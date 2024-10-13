import time
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from math import sqrt, inf, degrees, radians

class SkillLibrary:
    def __init__(self):
        super(SkillLibrary, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        

        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        

        return pose.pose

   

    def move(self, goal_type, goal, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=5.0, constraints=None):
        arm_group = self.arm_group

        # Set Parameters
        arm_group.set_max_velocity_scaling_factor(vel)
        arm_group.set_max_acceleration_scaling_factor(accel)
        arm_group.set_num_planning_attempts(attempts)
        arm_group.set_planning_time(time)
    
        

        if goal_type == 'pose':
            arm_group.clear_pose_targets()
            arm_group.set_goal_position_tolerance(tolerance)

            if constraints is not None:
                arm_group.set_path_constraints(constraints)
            
            arm_group.set_pose_target(goal)

            # Plan and Execute
            (success, plan, planning_time, error_code) = arm_group.plan()
            if success:
                print("Planning was successful")
                print(f"Panning time: {planning_time}")
                print("Executing the plan")
                joint_positions = plan.joint_trajectory.points[-1].positions

                print("Last Planning Angles: ", [degrees(joint_positions[i]) for i in range(len(joint_positions))])
                print("Planning size: ", len(plan.joint_trajectory.points))
        
                success = arm_group.execute(plan, wait=True)
                arm_group.stop()
                arm_group.clear_pose_targets()
            else:
                print("Planning failed")

        elif goal_type == 'joint':
            # Get the current joint positions
            joint_positions = arm_group.get_current_joint_values()

            # Set the goal joint tolerance
            self.arm_group.set_goal_joint_tolerance(tolerance)

            # Set the joint target configuration
            joint_positions[0] = goal[0]
            joint_positions[1] = goal[1]
            joint_positions[2] = goal[2]
            joint_positions[3] = goal[3]
            joint_positions[4] = goal[4]
            joint_positions[5] = goal[5]
            arm_group.set_joint_value_target(joint_positions)

            # Plan & Execute
            (success, plan, planning_time, error_code) = arm_group.plan()
            if success:
                print("Planning Successful.")
                print(f"Planning time: {planning_time}")
                print("Executing Plan...")
                success = arm_group.execute(plan, wait=True)
                arm_group.stop()

        elif goal_type == 'path':
            # Clear old pose targets
            arm_group.clear_pose_targets()

            # Clear max cartesian speed
            arm_group.clear_max_cartesian_link_speed()

            # Set the tolerance
            arm_group.set_goal_position_tolerance(tolerance)

            # Set the trajectory constraint if one is specified
            if constraints is not None:
                arm_group.set_path_constraints(constraints)

            eef_step = 0.01
            jump_threshold = 0.0
            (plan, fraction) = arm_group.compute_cartesian_path(goal, eef_step, jump_threshold)
            success = arm_group.execute(plan, wait=True)
            arm_group.stop()

        elif goal_type == 'gripper':
            # We only have to move this joint because all others are mimic!
            gripper_joint = self.robot.get_joint(self.gripper_joint_name)
            gripper_max_absolute_pos = gripper_joint.max_bound()
            gripper_min_absolute_pos = gripper_joint.min_bound()
            success = gripper_joint.move(goal * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)

        else:
            rospy.ERROR("Invalid goal type")

        return success
    

    def reach_named_position(self, target):
        arm_group = self.arm_group
        
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # Plan the trajectory
        (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return arm_group.execute(trajectory_message, wait=True)
    

    def grasp_object(self, goal_type, goal, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=5.0, constraints= None):
        
        # # # Move to grasp scan position
        # success = self.reach_named_position("overlook")
        # if not success:
        #     return False
        
        # TODO : Calculate the gipper pose based on the object
        # Gripper pose for the pre-grasp (Need improvement)

        # success = self.move('gripper', 0.65)
        # if not success:
        #     return False
        
        # Move to grasp position
        success = self.move(goal_type, goal, tolerance, vel, accel, attempts, time, constraints)
        if not success:
            return False
        
        # # Close the gripper
        # success = self.move('gripper', 0.85)
        # if not success:
        #     return False
        
        # success = self.reach_named_position("overlook")
        # if not success:
        #     return False


    





        