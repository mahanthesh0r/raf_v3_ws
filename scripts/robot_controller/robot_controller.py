import rospy

from raf_v3.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from raf_v3.srv import JointCommand, JointCommandRequest, JointCommandResponse
from raf_v3.srv import JointWaypointsCommand, JointWaypointsCommandRequest, JointWaypointsCommandResponse
from raf_v3.srv import GripperCommand, GripperCommandRequest, GripperCommandResponse

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .base import RobotController

class KinovaRobotController(RobotController):
    def __init__(self):
        # Do nothing
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        self.acq_pos = [0.004, 0.063, -1.100, -0.004, -1.980, 1.566, 4.814377930122582]
        self.transfer_pos = [3.9634926355200855, 5.7086929905176556, 4.912630464851094, 4.31408101511415, 4.877527871154977, 5.429743910562832, 3.8112093559638285]
        self.feed_joint_pose = [1.010, -0.034, -1.849, 1.192, -1.976, -0.356]

    def reset(self):
        self.move_to_acq_pose()

    def move_to_feed_pose(self):
        print('Moving to feed pose')
        self.set_joint_position(self.feed_joint_pose)

    def move_to_pose(self, pose):
        print("Calling set_pose with pose: ", pose)
        rospy.wait_for_service('/my_gen3/set_pose')
        try:
            move_to_pose = rospy.ServiceProxy('/my_gen3/set_pose', PoseCommand)
            resp1 = move_to_pose(pose, self.DEFAULT_FORCE_THRESHOLD)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_joint_position(self, joint_position, mode = "POSITION"):
        print("Calling set_joint_positions with joint_position: ", joint_position)
        rospy.wait_for_service('set_joint_position')
        try:
            move_to_joint_position = rospy.ServiceProxy('set_joint_position', JointCommand)
            resp1 = move_to_joint_position(mode, joint_position, 100.0)
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

    def set_gripper(self, gripper_target):
        print("Calling set_gripper with gripper_target: ", gripper_target)
        rospy.wait_for_service('my_gen3/set_gripper')
        try:
            set_gripper = rospy.ServiceProxy('/my_gen3/set_gripper', GripperCommand)
            resp1 = set_gripper(gripper_target)
            return resp1.success
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def move_to_acq_pose(self):
        print('Moving to acq pose')
        self.set_joint_position(self.acq_pos)

    def move_to_transfer_pose(self):
        print('Moving to transfer pose')
        self.set_joint_position(self.transfer_pos)

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    robot_controller = KinovaRobotController()

    # input('Press enter to move to acquisition position...')
    # robot_controller.move_to_acq_pose()

    # input('Press enter to move to transfer position...')
    # robot_controller.move_to_transfer_pose()
    

    robot_controller.set_gripper(0.0)
    # robot_controller.set_joint_position([6.26643082812968, 5.964520505888411, 3.226885713821761, 4.113400641700101, 0.44228980435708964, 6.056389443484003, 1.5805738564210134])

    # input('Press enter to reset the robot...')
    # robot_controller.reset()
        