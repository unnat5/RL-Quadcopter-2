import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
             target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 7
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.temp = 10.0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # task.sim.pose (the position of the quadcopter in (x,y,z)
        # task.sim.v (the velocity of the quadcopter in (x,y,z) dimension
        # task.sim.angular_v (radians/seconds for each euler angle)
        # Reward for landing softly.
        
        # Lets make problem in terms of z axis first.
        #print(self.sim.pose[2])
        
        #reward = -(abs((self.sim.pose[2]+self.sim.v[2]+(abs(self.sim.pose[2]-self.temp)**2))-3*abs(np.min((self.sim.pose[2]- 5),0)) *self.sim.pose[2]   )/(self.sim.runtime))
        #reward = -(abs((self.sim.pose[2]+7*abs(self.sim.v[2])+(6*abs(self.sim.pose[2]-self.temp))-3*abs(np.min((self.sim.pose[2]- 5),0))*self.sim.pose[2])))+self.sim.time
        #reward = np.clip(-(abs((self.sim.pose[2]+abs(self.sim.v[2])+(abs(self.sim.pose[2]-self.temp))-3*abs(np.min((self.sim.pose[2]- 5),0))*self.sim.pose[2])))+self.sim.time,-1,1)
        #else :
            #reward = -np.clip( ((self.sim.v[2] + self.sim.pose[2]*0.0001)/(2*self.sim.runtime)),0,1.0)
            #reward = 1.0 - np.clip((0.4*abs((0.2*(self.sim.pose[2])/(0.8*((abs(self.sim.pose[2]-self.temp)))+0.001) + abs(self.sim.pose[0])+abs(self.sim.pose[1])))) + 1.5*(self.sim.runtime),-1.0,1.0)
        #reward =np.tanh(self.sim.pose[2] -0.7*(abs(self.sim.pose[2]-self.target_pos[2])**2)) 
        
        reward =10*self.sim.v[2]**3+0.7*((self.sim.pose[2]-self.target_pos[2])) +(self.temp - self.sim.pose[2])
        #reward = np.clip(reward,-1,1)
        
        #print('t',self.temp)
        #print('p',self.sim.pose[2])
        self.temp = self.sim.pose[2]
        reward = np.tanh(reward)
       
        #print(self.sim.v[2])
            

        
        
        #print(reward)
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            velocity_z = list([self.sim.v[2]])
            position = list(self.sim.pose)
            new_state = position + velocity_z
            new_state = np.array(new_state)
            #print(new_state)
            pose_all.append(new_state)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        velocity_z = list([self.sim.v[2]])
        position = list(self.sim.pose)
        new_state = position + velocity_z
        new_state = np.array(new_state)
        
        state = np.concatenate([new_state] * self.action_repeat)
        #print('ss',self.sim.pose.shape)
        return state