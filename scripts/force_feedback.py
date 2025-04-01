import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kortex_driver.msg import BaseCyclic_Feedback

class ForceFeedback:
    def __init__(self):
        rospy.init_node('force_feedback', anonymous=True)
        self.force_x = []
        self.force_y = []
        self.force_z = []
        self.time_stamps = []
        self.start_time = rospy.get_time()

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True)
        self.line_x, = self.ax1.plot([], [], 'r-', label='Force X')
        self.line_y, = self.ax2.plot([], [], 'g-', label='Force Y')
        self.line_z, = self.ax3.plot([], [], 'b-', label='Force Z')

        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(-20, 20)
        self.ax1.set_ylabel('Force X (N)')
        self.ax1.set_title('Real-Time Force Feedback')
        self.ax1.legend()

        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(-20, 20)
        self.ax2.set_ylabel('Force Y (N)')
        self.ax2.legend()

        self.ax3.set_xlim(0, 10)
        self.ax3.set_ylim(-20, 20)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Force Z (N)')
        self.ax3.legend()

        self.sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.callback)

    def callback(self, msg):
        current_time = rospy.get_time() - self.start_time
        self.time_stamps.append(current_time)
        self.force_x.append(msg.base.tool_external_wrench_force_x)
        self.force_y.append(msg.base.tool_external_wrench_force_y)
        self.force_z.append(msg.base.tool_external_wrench_force_z)

        # Keep only the last 10 seconds of data
        if current_time > 10:
            self.time_stamps.pop(0)
            self.force_x.pop(0)
            self.force_y.pop(0)
            self.force_z.pop(0)

    def update_plot(self, frame):
        if not self.time_stamps:
            return self.line_x, self.line_y, self.line_z

        self.line_x.set_data(self.time_stamps, self.force_x)
        self.line_y.set_data(self.time_stamps, self.force_y)
        self.line_z.set_data(self.time_stamps, self.force_z)

        self.ax1.set_xlim(max(0, self.time_stamps[-1] - 10), self.time_stamps[-1])
        self.ax2.set_xlim(max(0, self.time_stamps[-1] - 10), self.time_stamps[-1])
        self.ax3.set_xlim(max(0, self.time_stamps[-1] - 10), self.time_stamps[-1])

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        return self.line_x, self.line_y, self.line_z

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update_plot, blit=True, interval=100)
        plt.show()

if __name__ == "__main__":
    force_feedback = ForceFeedback()
    force_feedback.run()
    rospy.spin()