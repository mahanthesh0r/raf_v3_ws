import rospy

from raf_v3.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from raf_v3.srv import JointCommand, JointCommandRequest, JointCommandResponse
from raf_v3.srv import JointWaypointsCommand, JointWaypointsCommandRequest, JointWaypointsCommandResponse
from raf_v3.srv import GripperCommand, GripperCommandRequest, GripperCommandResponse

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .base import RobotController
import time
import yaml
import raf_utils

class KinovaRobotController(RobotController):
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.isTable = config['on_table']
        # Do nothing
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        # old overlook position
        self.acq_pos = [-0.000, 0.000, -1.571, 0.000, -1.571, 1.571]
        # new overlook position (camera parallel to table)
        # self.acq_pos = [6.281806497, 0.02157227, 4.491866629, 6.268070756, 4.919193043, 1.574095]  
        #self.overlook_table = [0, 0.471518151, 5.247698915, 0, 4.650045819, 6.28319]
        self.overlook_table = [6.280689486, 0.534978322, 4.866274661, 0.0016231562, 5.094354287, 0]

        # camera parallel to table but raised
        #self.acq_pos = [0.001047198, 6.282295189, 4.736963216, 6.267442437, 4.650464699, 1.56714859]
        self.transfer_pos = [3.9634926355200855, 5.7086929905176556, 4.912630464851094, 4.31408101511415, 4.877527871154977, 5.429743910562832, 3.8112093559638285]
        #self.feed_joint_pose = [1.010, -0.034, -1.849, 1.192, -1.976, -0.356]
        #self.feed_joint_pose = [0.638, 0.0083, -1.711, 1.211, -2.090, -0.407]
        self.feed_joint_pose = [0.979932562, 0.255900175, 4.526441602, 1.41794039, 4.205109033, 5.853851764]
        self.cup_joints = [4.446086643, 1.13195074, 5.0282936, 3.72753459, 2.099927796, 1.948415764] 
        self.on_table_cup_scan = [5.547581387, 1.19961715, 4.673782297, 1.782295326, 5.002724501, 5.011328975]

        # old sip pose
        #self.sip_pose = [0.914517621, 0.205128547, 4.589919227, 4.33155814, 2.030097173, 2.809875376]
        self.sip_pose_tall = [1.01763167, 0.14025466, 4.589203642, 4.305814531, 2.024267773, 2.806053105]
        self.sip_pose_short = [1.01649721, 0.201986954, 4.433816978, 4.221550035, 1.925726483, 2.595252238]
        self.on_table_sip_pose_tall = [0.260630017, 0.780912667, 4.666626447, 1.857344483, 5.160694252, 5.453909566]
        self.on_table_sip_pose_short = [0.260472938, 0.863274755, 4.654688395, 1.898534254, 5.133379849, 5.35519374]

        #self.multi_bite_transfer = [0.74991562, 0.052394784, 4.51097798, 1.31203381, 4.214306918, 5.876802844]
        self.multi_bite_tall_transfer = [1.02595689, 6.262712595, 4.498324348, 1.22686174, 4.219717439, 5.882195912]
        self.single_bite_tall_transfer = [0.928375536, 6.24190827, 4.276196294, 1.43356109, 4.198843301, 5.842297685]
        self.single_bite_short_transfer = [0.92786939, 6.27760025, 4.205126487, 1.37528454, 4.217099445, 5.722323752]
        self.multi_bite_short_transfer = [1.0253809, 6.281038552, 4.418143922, 1.17991239, 4.255583955, 5.779553098]
        self.single_bite_tall_transfer_on_table = [0.38627627, 0.660903828, 4.72537423, 1.987842752, 5.213036676, 5.530512067]
        self.single_bite_short_transfer_on_table = [0.385909751, 0.756111539, 4.679157911, 2.052472294, 5.153032257, 5.386993643]
        self.single_bite_trasnfer = config['joint_positions']['bite_transfer']
        self.multi_bite_transfer = config['joint_positions']['multi_bite_transfer']
        self.cup_feed_pose = config['joint_positions']['cup_feed_pose']

        self.pre_calibration_pose = [6.14757322, 0.608613763, 4.6216319, 1.64115055, 4.807526878, 5.575436842]
        self.pre_calibration_pose_degrees = [352.23,34.871,264.8,94.031,275.415,319.449]
    async def reset(self):
        await self.move_to_acq_pose()
        #await self.setting_gripper_value(0.0)
        
    async def move_to_cup_joint(self):
        print('Moving to cup joint')
        if self.isTable:
            status = await self.set_joint_position(self.on_table_cup_scan)
        else:
            status = await self.set_joint_position(self.cup_joints)
        return status

    async def move_to_feed_pose(self, height='TALL'):
        print('Moving to feed pose', height)
        if height == 'TALL':
            if self.isTable:
               await self.set_joint_position(self.single_bite_tall_transfer_on_table)
            else:
                await self.set_joint_position(self.single_bite_tall_transfer)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.single_bite_short_transfer_on_table)
            else:
                await self.set_joint_position(self.single_bite_short_transfer)
        else:
            #Custom height for feed pose using gravity compensation mode
            joint_velocities = raf_utils.getJointVelocity(self.pre_calibration_pose_degrees, self.single_bite_trasnfer, 3.5)
            await self.set_joint_velocity(joint_velocities)

    async def move_to_sip_pose(self, height='TALL'):
        print('Moving to sip pose')
        if height == 'TALL':
            if self.isTable:
                await self.set_joint_position(self.on_table_sip_pose_tall)
            else:
                await self.set_joint_position(self.sip_pose_tall)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.on_table_sip_pose_short)
            else:
                await self.set_joint_position(self.sip_pose_short)
        else:
            #Custom height for sip pose using gravity compensation mode
            await self.set_joint_position(self.cup_feed_pose)

    async def move_to_multi_bite_transfer(self, height='TALL'):
        print('Moving to multi bite transfer', height)
        if height == 'TALL':
            if self.isTable:
                await self.set_joint_position(self.single_bite_tall_transfer_on_table)
            else:
                await self.set_joint_position(self.multi_bite_tall_transfer)
        elif height == 'SHORT':
            if self.isTable:
                await self.set_joint_position(self.single_bite_short_transfer_on_table)
            else:
                await self.set_joint_position(self.multi_bite_short_transfer)
        else:
            #Custom height for multi bite transfer using gravity compensation
            joint_velocities = raf_utils.getJointVelocity(self.pre_calibration_pose_degrees, self.multi_bite_transfer, 3.5)
            await self.set_joint_velocity(joint_velocities)
            

    async def move_to_pose(self, pose):
        print("Calling set_pose with pose: ", pose)
        rospy.wait_for_service('/my_gen3/set_pose')
        try:
            move_to_pose = rospy.ServiceProxy('/my_gen3/set_pose', PoseCommand)
            resp1 = move_to_pose(pose, self.DEFAULT_FORCE_THRESHOLD)
            print("Response: ", resp1)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    async def set_joint_position(self, joint_position, mode = "POSITION"):
        print("Calling set_joint_positions with joint_position: ", joint_position)
        rospy.wait_for_service('my_gen3/set_joint_position')
        try:
            move_to_joint_position = rospy.ServiceProxy('/my_gen3/set_joint_position', JointCommand)
            resp1 = move_to_joint_position(mode, joint_position, 100.0)
            print("Response: ", resp1)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    async def set_joint_velocity(self, joint_velocity, mode = "VELOCITY"):
        print("Calling set_joint_velocity with joint_velocity: ", joint_velocity)
        rospy.wait_for_service('my_gen3/set_joint_velocity')
        try:
            set_joint_velocity = rospy.ServiceProxy('/my_gen3/set_joint_velocity', JointCommand)
            resp1 = set_joint_velocity(mode, joint_velocity, 3.5)
            print("Response: ", resp1)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_joint_waypoints(self, joint_waypoints):
        print("Calling set_joint_waypoints with joint_waypoints")
        rospy.wait_for_service('set_joint_waypoints')
        try:
            move_to_joint_waypoints = rospy.ServiceProxy('set_joint_waypoints', JointWaypointsCommand)

            target_waypoints = JointTrajectory()
            target_waypoints.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            
            for waypoint in joint_waypoints:
                point = JointTrajectoryPoint()
                point.positions = waypoint
                target_waypoints.points.append(point)
            
            resp1 = move_to_joint_waypoints(target_waypoints, 100.0)
            return resp1.success
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    async def setting_gripper_value(self, gripper_value):
        await self.set_gripper(gripper_value)
        rospy.sleep(1)

    async def set_gripper(self, gripper_target):
        print("Calling set_gripper with gripper_target: ", gripper_target)
        rospy.wait_for_service('my_gen3/set_gripper')
        try:
            set_gripper = rospy.ServiceProxy('/my_gen3/set_gripper', GripperCommand)
            resp1 = set_gripper(gripper_target)
            print("Gripper response: ", resp1)
            return resp1.success
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    async def move_to_acq_pose(self):
        print('Moving to acq pose')
        if self.isTable:
            await self.set_joint_position(self.overlook_table)
        else:
            await self.set_joint_position(self.acq_pos)

    async def move_to_transfer_pose(self):
        print('Moving to transfer pose')
        await self.set_joint_position(self.feed_joint_pose)

    async def move_to_pre_calibration_pose(self):
        print('Moving to pre calibration pose')
        await self.set_joint_position(self.pre_calibration_pose)

    

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    robot_controller = KinovaRobotController()

    # input('Press enter to move to acquisition position...')
    # robot_controller.move_to_acq_pose()

    # input('Press enter to move to transfer position...')
    # robot_controller.move_to_transfer_pose()
    

    
    # robot_controller.set_joint_position([6.26643082812968, 5.964520505888411, 3.226885713821761, 4.113400641700101, 0.44228980435708964, 6.056389443484003, 1.5805738564210134])

    # input('Press enter to reset the robot...')
    # robot_controller.reset()
        