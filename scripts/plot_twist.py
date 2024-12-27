import rospy
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import kortex_driver.msg

class TwistPlotter:
    def __init__(self):
        rospy.init_node('twist_plotter', anonymous=True)

        # Subscribers
        self.twist_sub = rospy.Subscriber('/cmd_vel', kortex_driver.msg.TwistCommand, self.twist_callback)

        # Data storage
        self.linear_x_data = []
        self.linear_y_data = []
        self.time_data = []

        # Initialize plot
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], label='linear_x')
        self.line2, = self.ax.plot([], [], label='linear_y')
        self.ax.legend()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Velocity (m/s)')

        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=100)

    def twist_callback(self, msg):
        current_time = rospy.get_time()
        self.time_data.append(current_time)
        self.linear_x_data.append(msg.twist.linear_x)
        self.linear_y_data.append(msg.twist.linear_y)

        # Keep only the last 10 seconds of data
        if self.time_data[-1] - self.time_data[0] > 10:
            self.time_data.pop(0)
            self.linear_x_data.pop(0)
            self.linear_y_data.pop(0)

    def init_plot(self):
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        return self.line1, self.line2

    def update_plot(self, frame):
        self.line1.set_data(self.time_data, self.linear_x_data)
        self.line2.set_data(self.time_data, self.linear_y_data)
        self.ax.set_xlim(self.time_data[0], self.time_data[-1])
        return self.line1, self.line2

    def run(self):
        plt.show()

if __name__ == "__main__":
    try:
        plotter = TwistPlotter()
        plotter.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass