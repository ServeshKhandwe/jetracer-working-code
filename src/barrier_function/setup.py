from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'barrier_function'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ? Ensure launch files are installed
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.py', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='HeshamMousa95@eng.asu.edu.eg',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "LLM_robot_controller = my_robot_controller.LLM_robot_controller:main",
            "safety_check_nonlinear_reachability= my_robot_controller.safety_check_nonlinear_reachability:main",
            "safe_LLM_controller= my_robot_controller.safe_LLM_controller:main"
        ],
    },
)