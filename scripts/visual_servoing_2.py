import rospy
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import kortex_driver.msg
from raf_v3.msg import VisualServoData

class VisualServoingRobot:
    def __init__(self):
        rospy.init_node('visual_servoing_robot', anonymous=True)

        # Publishers
        self.velocity_pub = rospy.Publisher('/cmd_vel', kortex_driver.msg.TwistCommand, queue_size=10)

        

        # Desired position (center of the image)
        self.target_center = (810, 653)
        self.loop_rate = rospy.Rate(10)
        self.current_force_z = []
        self.velocity = 0.005
        self.depth_value = None
        self.object_center = None

        self.sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            kortex_driver.msg.BaseCyclic_Feedback,
            self.base_feedback_cb,
        )
        self.pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity",
            kortex_driver.msg.TwistCommand,
            queue_size=1,
        )

        rospy.loginfo("Visual Servoing Robot initialized.")

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

    def start(self):
        # Subscribe to the visual_servo_data topic
        self.visual_servo_sub = rospy.Subscriber('visual_servo_data', VisualServoData, self.visual_servo_callback)

    def visual_servo_callback(self, data):
        """Callback function to process the visual servo data."""
        self.object_center = (data.x, data.y)
        self.depth_value = data.depth
        rospy.loginfo(f"Received object center: {self.object_center}")
        self.perform_visual_servoing(self.object_center, self.depth_value)
    def perform_visual_servoing(self, object_center, depth_value):
        """Perform visual servoing based on the detected object's position."""
        print("Performing visual servoing...")
        cx, cy = object_center
        tx, ty = self.target_center

        # Compute the error
        error_x = cx - tx
        error_y = cy - ty

        # depth_value = self.depth_image[cy, cx] / 1000.0
        # print("Depth for pixel (", cx, ",", cy, "):", depth_value)
        

        # Generate velocity commands (P-controller)
        msg = kortex_driver.msg.TwistCommand()
        gain_x = 0.001  # Gain for linear velocity
        gain_z = 0.4 # Gain for angular velocity

        # Adjust velocities based on the error
        msg.twist.linear_x = -gain_x * error_y  # Move forward/backward
        msg.twist.linear_y = -gain_x * error_x  # Move left/right
        #msg.twist.angular.z = -gain_z * error_x  # Rotate to center object


        depth_value = round(depth_value, 3)
        if depth_value > 0.190:
            print("Move down Depth value:", depth_value)
            msg.twist.linear_z = -gain_z * (depth_value - 0.19)  # Move up/down
        elif depth_value < 0.200 and depth_value > 0.0:
            print("Move up Depth value:", depth_value)
            msg.twist.linear_z = gain_z * (depth_value - 0.2)
        elif depth_value == 0.0 or depth_value == float('nan'):
            print("NaN Stop Depth value:", depth_value)
            msg.twist.linear_z = 0.0
        else:
            print("Stop Depth value:", depth_value)
            msg.twist.linear_z = 0.0

        self.pub.publish(msg)

        # Publish the velocity command
        self.velocity_pub.publish(msg)

    
if __name__ == "__main__":
    try:
        cvs =  VisualServoingRobot()
        cvs.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Visual Servoing Robot node terminated.")
    finally:
        cv2.destroyAllWindows()