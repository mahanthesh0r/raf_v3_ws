import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import kortex_driver.msg
from rs_ros import RealSenseROS

class CustomVisualServoing:
    def __init__(self):
        rospy.init_node('custom_visual_servoing', anonymous=True)

        # Publishers
        self.velocity_pub = rospy.Publisher('/cmd_vel', kortex_driver.msg.TwistCommand, queue_size=10)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Desired position (center of the image)
        self.target_center = None
        self.image_width = None
        self.image_height = None
        self.loop_rate = rospy.Rate(10)
        self.current_force_z = []
        self.velocity = 0.005
        self.stop = False
        self.camera = RealSenseROS()

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

        rospy.loginfo("Custom Visual Servoing with Visualization initialized.")

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

    def start(self):
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        print("Subscribed to /camera/color/image_raw")
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)


    def depth_callback(self, data):
        """Process depth images."""
        
        try:
            # Convert the ROS Image message to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(f"Error in depth image processing: {e}")

    def image_callback(self, data):
        """Process images, visualize, and perform visual servoing."""
        
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

            # Get image dimensions
            self.image_height, self.image_width, _ = cv_image.shape
            # Image center
            self.target_center = (self.image_width // 2, self.image_height // 2)
            # Gripper center
            self.target_center = (766, 601)

            # Detect the blue ball
            object_center = self.detect_blue_ball(cv_image)

            # Draw the target center
            cv2.circle(cv_image, self.target_center, 10, (0, 255, 0), -1)  # Green for target

            if object_center and not self.stop:
                # Draw the detected object
                cv2.circle(cv_image, object_center, 10, (255, 0, 0), -1)  # Blue for detected object

                # Perform visual servoing
                self.perform_visual_servoing(object_center)

                # Draw the error as a line
                cv2.line(cv_image, self.target_center, object_center, (0, 0, 255), 2)  # Red for error
            else:
                rospy.logwarn("No blue ball detected.")

            # Display the image
            cv2.imshow("Visual Servoing - Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")

    def detect_blue_ball(self, image):
        """Detect a blue ball in the image and return its center."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_orange = (10, 100, 100)
        upper_orange = (25, 255, 255)
        
        # Create a mask for the orange color
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        return None

    def perform_visual_servoing(self, object_center):
        """Perform visual servoing based on the detected object's position."""
        cx, cy = object_center
        tx, ty = self.target_center

        # Compute the error
        error_x = cx - tx
        error_y = cy - ty

        depth_value = self.depth_image[cy, cx] / 1000.0
        print("Depth for pixel (", cx, ",", cy, "):", depth_value)
        

        # Generate velocity commands (P-controller)
        msg = kortex_driver.msg.TwistCommand()
        gain_x = 0.001  # Gain for linear velocity
        gain_z = 2  # Gain for angular velocity

        # Adjust velocities based on the error
        msg.twist.linear_x = -gain_x * error_y  # Move forward/backward
        msg.twist.linear_y = -gain_x * error_x  # Move left/right
        #msg.twist.angular.z = -gain_z * error_x  # Rotate to center object

        if depth_value > 0.350:
            print("Move down Depth value:", depth_value)
            msg.twist.linear_z = -gain_z * (depth_value - 0.25)  # Move up/down
        elif depth_value < 0.250 and depth_value > 0.0:
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
        cvs =  CustomVisualServoing()
        cvs.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Custom Visual Servoing node terminated.")
    finally:
        cv2.destroyAllWindows()
