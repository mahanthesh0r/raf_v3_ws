#ifndef GEN3_ROBOT_H
#define GEN3_ROBOT_H

// ros
#include <ros/ros.h>
#include <ros/console.h>

// kinova api
#include <client/TransportClientTcp.h>
#include <client/RouterClient.h>
#include <client/SessionManager.h>
#include <client_stubs/BaseClientRpc.h>
#include <client_stubs/BaseCyclicClientRpc.h>

// not sure if this is required
#include <client_stubs/ActuatorConfigClientRpc.h>

// c++
#include <stdexcept>
#include <limits>
#include <iostream>
#include <unistd.h>
#include <time.h>

// msg
#include <std_msgs/Bool.h>
#include <sensor_msgs/JointState.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "raf_v3/CartesianState.h"
#include "raf_v3/JointCommand.h"
#include "raf_v3/JointWaypointsCommand.h"
#include "raf_v3/PoseCommand.h"
#include "raf_v3/PoseWaypointsCommand.h"
#include "raf_v3/TwistCommand.h"
#include "raf_v3/GripperCommand.h"

#define TCP_PORT 10000

using namespace std;
namespace k_api = Kinova::Api;

// Maximum allowed waiting time during actions
constexpr auto TIMEOUT_DURATION = std::chrono::seconds{20};


class Controller
{
    public:
        Controller(ros::NodeHandle nh);
        virtual ~Controller();
        
    private:

        inline double degreesToRadians(double degrees);
        inline double radiansToDegrees(double radians);

        // ros services for joint position, pose and twise control
        bool setJointPosition(raf_v3::JointCommandRequest &request, raf_v3::JointCommandResponse &response);
        bool setJointWaypoints(raf_v3::JointWaypointsCommandRequest &request, raf_v3::JointWaypointsCommandResponse &response);
        bool setJointVelocity(raf_v3::JointCommandRequest &request, raf_v3::JointCommandResponse &response);
        bool setPose(raf_v3::PoseCommandRequest &request, raf_v3::PoseCommandResponse &response);
        bool setPoseWaypoints(raf_v3::PoseWaypointsCommandRequest &request, raf_v3::PoseWaypointsCommandResponse &response);
        bool setTwist(raf_v3::TwistCommandRequest &request, raf_v3::TwistCommandResponse &response);
        bool setGripper(raf_v3::GripperCommandRequest &request, raf_v3::GripperCommandResponse &response);

        ros::ServiceServer mSetJointPositionService;
        ros::ServiceServer mSetJointWaypointsService;
        ros::ServiceServer mSetJointVelocityService;
        ros::ServiceServer mSetPoseService;
        ros::ServiceServer mSetPoseWaypointsService;
        ros::ServiceServer mSetTwistService;
        ros::ServiceServer mSetGripperService;

        std::atomic<bool> mWatchdogActive;

        // joint and cartesian state publisher
        ros::Timer mRobotStateTimer;
        ros::Publisher mJointStatePub;
        ros::Publisher mCartesianStatePub;
        void publishState(const ros::TimerEvent& event);

        ros::Subscriber mTareFTSensorSub;
        std::atomic<bool> mTareFTSensor;
        std::vector <double> mZeroFTSensorValues;
        std::vector <double> mFTSensorValues;
        void tareFTSensorCallback(const std_msgs::Bool& msg);

        ros::Subscriber mEStopSub;
        void eStopCallback(const std_msgs::Bool& msg);

        std::atomic<bool> mUpdateForceThreshold;
        std::vector <double> mForceThreshold;
        std::vector <double> mNewForceThreshold;

        std::string m_username;
        std::string m_password;
        int m_api_session_inactivity_timeout_ms;
        int m_api_connection_inactivity_timeout_ms;

        // Kortex Api objects
        std::string m_ip_address;
        k_api::TransportClientTcp*  m_tcp_transport;
        k_api::RouterClient* m_tcp_router;
        k_api::SessionManager* m_tcp_session_manager;

        k_api::Base::BaseClient* mBase;
        k_api::BaseCyclic::BaseCyclicClient* mBaseCyclic;
        k_api::BaseCyclic::Command mBaseCommand;
        k_api::BaseCyclic::Feedback mLastFeedback;

        k_api::ActuatorConfig::ActuatorConfigClient* mActuatorConfig;
        k_api::Base::ServoingModeInformation mServoingMode;
        k_api::ActuatorConfig::ControlModeInformation mControlModeMessage;

        // Unused - For frequency monitoring
        int64_t now = 0;
        int64_t last = 0;
};

#endif

