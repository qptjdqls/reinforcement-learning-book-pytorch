import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Actor(nn.Module):
    ''' A2C 액터 신경망 '''
    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()
        self.action_bound = action_bound

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(16, action_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(16, action_dim),
            nn.Softplus()
        )

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        std = self.std(x)

        # 평균값을 [-action_bound, action_bound] 범위로 조정
        mu = mu * self.action_bound

        return [mu, std]


class Critic(nn.Module):
    ''' A2C 크리틱 신경망 '''
    def __init__(self, state_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.v = nn.Linear(16, 1)

    def forward(self, state):
        x = self.fc(state)
        v = self.v(x)
        return v
    

class A2Cagent:
    ''' A2C 에이전트 클래스 '''
    def __init__(self, env):
        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]
        # 표준편차의 최솟값과 최대값 설정
        self.std_bound = [1e-2, 1.0]

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim)

        # 옵티마이저 설정
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE
        )
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE
        )

        #  에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []

    def log_pdf(self, mu, std, action):
        ''' 로그-정책 확률밀도함수 '''
        std = torch.clip(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action-mu) ** 2 / var \
            - 0.5 * torch.log(var*2*np.pi)
        return torch.sum(log_policy_pdf, 1, keepdims=True)

    @ torch.no_grad()
    def get_action(self, state):
        ''' 액터 신경망에서 행동 샘플링 '''
        mu_a, std_a = self.actor(state)
        std_a = torch.clip(std_a, self.std_bound[0], self.std_bound[1])
        # action = torch.normal(mu_a, std_a, size=self.action_dim)
        action = torch.normal(mu_a, std_a)
        return action
    
    def actor_learn(self, states, actions, advantages):
        ''' 액터 신경망 학습 '''
        # 정책 확률밀도함수
        mu_a, std_a = self.actor(states)
        log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

        # 손실함수
        loss_policy = log_policy_pdf * advantages
        loss = torch.sum(-loss_policy)

        # 그래디언트
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def critic_learn(self, states, td_targets):
        ''' 크리틱 신경망 학습 '''
        td_hat = self.critic(states)
        loss = torch.mean(torch.square(td_targets - td_hat))

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

    @ torch.no_grad()
    def td_target(self, rewards, next_v_values, dones):
        '''  시간차 타깃 계산 '''
        next_v_values = next_v_values.numpy()
        y_i = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):
            if dones[i]:
                y_i[i] = rewards[i]
            else:
                y_i[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return y_i

    def load_weights(self, path):
        ''' 신경망 파라미터 로드 '''
        self.actor.load_state_dict(torch.load(path + 'pendulum_actor.pth'))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(path + 'pendulum_critic.pth'))
        self.critic.eval()

    def unpack_batch(self, batch):
        '''  배치에 저장된 데이터 추출 '''
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
         
        return unpack

    def train(self, max_episode_num):
        ''' 에이전트 학습 '''
        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):
            # 배치 초기화
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [], [], [], [], []
            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()
        
            while not done:
                # 학습 가시화
                # self.env.render()

                # 행동 샘플링
                action = self.get_action(torch.tensor(state, dtype=torch.float32))
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # shape 변환
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                done = np.reshape(done, [1, 1])
                # 학습용 보상 계산
                train_reward = (reward + 8) / 8

                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue
                    
                # 배치가 채워지면 학습 진행
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # 배치 비움
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                    [], [], [], [], []

                # 시간차 타깃 계산
                next_v_values = self.critic(torch.tensor(next_states, dtype=torch.float32))
                td_targets = self.td_target(train_rewards, next_v_values, dones)

                # 크리틱 신경망 업데이트
                self.critic_learn(
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(td_targets, dtype=torch.float32)
                )

                # 어드밴티지 계산
                v_values = self.critic(torch.tensor(states, dtype=torch.float32))
                next_v_values = self.critic(torch.tensor(next_states, dtype=torch.float32))
                train_rewards = torch.tensor(train_rewards, dtype=torch.float32)
                advantages = train_rewards + self.GAMMA * next_v_values - v_values
                
                # 액터 신경망 업데이트
                self.actor_learn(
                    torch.tensor(states, dtype=torch.float32), 
                    torch.tensor(actions, dtype=torch.float32), 
                    torch.tensor(advantages, dtype=torch.float32)
                )

                # 상태 업데이트
                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            # 에피소드마다 결과 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                torch.save(self.actor.state_dict(), './save_weights/pendulum_actor.pth')
                torch.save(self.critic.state_dict(), './save_weights/pendulum_critic.pth')
            
        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    # 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        # plt.show()
        plt.savefig('result.png')


    
