"""Simple abstract base class for robot wrappers."""

from abc import ABC, abstractmethod
from typing import Any


class RobotBase(ABC):
    """Minimal contract that concrete robot backends must follow."""

    @abstractmethod
    def get_joint_positions(self) -> Any:
        """Return the current joint positions."""

    @abstractmethod
    def get_joint_velocities(self) -> Any:
        """Return the current joint velocities."""

    @abstractmethod
    def get_joint_external_torques(self) -> Any:
        """Return the measured external joint torques."""

    @abstractmethod
    def get_tau_J(self) -> Any:
        """Return the joint torques."""

    @abstractmethod
    def get_ee_pose(self) -> Any:
        """Return the end-effector pose."""

    @abstractmethod
    def get_ee_T(self) -> Any:
        """Return the end-effector transformation matrix."""

    @abstractmethod
    def get_ee_wrench(self) -> Any:
        """Return the end-effector wrench."""

    @abstractmethod
    def get_base_wrench(self) -> Any:
        """Return the base wrench."""

    @abstractmethod
    def control(self, controller_type: str, action: Any, **kwargs: Any) -> Any:
        """Send a control command to the robot backend."""

    def close(self) -> None:
        """Optional cleanup hook for shutting down the backend connection."""

        return None
