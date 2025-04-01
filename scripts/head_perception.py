from rs_ros import RealSenseROS
import raf_utils as utils
import numpy as np
import cv2
import face_alignment
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import kortex_driver.msg

class HeadPerception:
    def __init__(self):
        rospy.init_node('head_perception')

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.mouth_center_publisher = rospy.Publisher('/mouth_center', Point, queue_size=10)
        self.mouth_open_publisher = rospy.Publisher('/mouth_open', Bool, queue_size=10)
        # Publishers
        self.velocity_pub = rospy.Publisher('/cmd_vel', kortex_driver.msg.TwistCommand, queue_size=10)
        self.camera = RealSenseROS()
        self.target_center = None
        self.loop_rate = rospy.Rate(10)
        self.current_force_z = []
        self.velocity = 0.005
        self.stop = False
        self.target_center = (811, 656)
        self.isMouthOpen = False
        self.soft_eStop = False

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

    def base_feedback_cb(self, msg):
        self.current_force_z.append(msg.base.tool_external_wrench_force_z)
        if len(self.current_force_z) > 25:
            self.current_force_z.pop(0)

        if abs(msg.base.tool_external_wrench_force_x) > 20.0:
            print("Force detected")
            self.soft_eStop = True


    def run(self):
        while not rospy.is_shutdown():
            camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

            if camera_color_data is None or camera_info_data is None or camera_depth_data is None:
                print("No data received")
                continue

            preds = self.fa.get_landmarks(camera_color_data)
            if preds is None:
                print("No face detected")
                continue

            preds = preds[0]
            preds_3d = []

            # visualize the landmarks
            for pred in preds[48:68]:
                validity, pred_3d = utils.pixel2World(camera_info_data, int(pred[0]), int(pred[1]), camera_depth_data, box_width=5)
                if validity:
                    preds_3d.append(pred_3d)
                    cv2.circle(camera_color_data, (int(pred[0]), int(pred[1])), 2, (0, 255, 0), -1)

            if len(preds_3d) == 0:
                print("No valid 3D landmarks")
                continue

            preds_3d = np.array(preds_3d)
            mouth_center_3d = np.mean(preds_3d, axis=0)

            mouth_center_3d_pixel = utils.world2Pixel(camera_info_data, mouth_center_3d[0], mouth_center_3d[1], mouth_center_3d[2])
            cv2.circle(camera_color_data, (int(mouth_center_3d_pixel[0]), int(mouth_center_3d_pixel[1])), 2, (0, 0, 255), -1)

            mouth_center_msg = Point()
            mouth_center_msg.x = mouth_center_3d[0]
            mouth_center_msg.y = mouth_center_3d[1]
            mouth_center_msg.z = mouth_center_3d[2]

            

            self.mouth_center_publisher.publish(mouth_center_msg)

            lipDist = np.sqrt((preds[66][0] - preds[62][0]) ** 2 + (preds[66][1] - preds[62][1]) ** 2)

            lipThickness = float(np.sqrt((preds[51][0] - preds[62][0]) ** 2 + (preds[51][1] - preds[62][1]) ** 2) / 2) + \
                           float(np.sqrt((preds[57][0] - preds[66][0]) ** 2 + (preds[57][1] - preds[66][1]) ** 2) / 2)

            if lipDist >= 1.5 * lipThickness:
                self.mouth_open_publisher.publish(True)
                self.isMouthOpen = True
                self.stop = False
            else:
                self.mouth_open_publisher.publish(False)
                self.isMouthOpen = False
                self.stop = True

            if self.isMouthOpen and not self.stop and not self.soft_eStop:
                # Perform visual servoing
                self.perform_visual_servoing(camera_color_data, camera_depth_data, mouth_center_3d_pixel)
                cv2.line(camera_color_data, self.target_center, (int(mouth_center_3d_pixel[0]), int(mouth_center_3d_pixel[1])), (0, 0, 255), 2)
            else:
                self.stop_visual_servoing()

            # visualize the landmarks
            cv2.imshow("Landmarks", camera_color_data)
            cv2.waitKey(1)

            # add signal handler to stop the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to break
                break

    def perform_visual_servoing(self, camera_color_data, camera_depth_data, mouth_center):
        """Perform visual servoing based on the detected object's position."""
        cx, cy = self.target_center
        tx, ty = mouth_center[0], mouth_center[1]

        # Compute the error
        error_x = cx - tx
        error_y = cy - ty

        depth_value = camera_depth_data[int(cy), int(cx)] / 1000.0

        # Generate velocity commands (P-controller)
        msg = kortex_driver.msg.TwistCommand()
        gain_x = 0.0002 # Gain for linear velocity
        slow_gain_x = 0.0001
        gain_z = 0.5  # Gain for angular velocity

        # Adjust velocities based on the error
        #msg.twist.linear_x = -gain_x * error_y  # Move forward/backward
        msg.twist.linear_y = -gain_x * error_x  # Move left/right
        msg.twist.linear_z = gain_x * error_y  # Move up/down
        print("Error x:", error_x, "Error y:", error_y, "Depth value:", depth_value)


        

        if depth_value > 0.350:
            print("Move down Depth value:", depth_value)
            msg.twist.linear_x = -gain_z * (depth_value - 0.25)  # Move up/down
        elif depth_value < 0.250 and depth_value > 0.0:
            print("Move up Depth value:", depth_value)
            msg.twist.linear_x = gain_z * (depth_value - 0.2)
        elif depth_value == 0.0 or depth_value == float('nan'):
            print("NaN Stop Depth value:", depth_value)
            msg.twist.linear_x = 0.0
        else:
            print("Stop Depth value:", depth_value)
            msg.twist.linear_x = 0.0

        self.pub.publish(msg)

        # Publish the velocity command
        self.velocity_pub.publish(msg)

    
    def stop_visual_servoing(self):
        msg = kortex_driver.msg.TwistCommand()
        msg.twist.linear_x = 0.0
        msg.twist.linear_y = 0.0
        msg.twist.linear_z = 0.0
        self.pub.publish(msg)
        self.velocity_pub.publish(msg)

        self.loop_rate.sleep()

if __name__ == '__main__':
    head_perception = HeadPerception()
    head_perception.run()
