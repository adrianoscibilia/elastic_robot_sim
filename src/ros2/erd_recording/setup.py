from setuptools import find_packages, setup
from glob import glob

package_name = "erd_recording"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config/lab", glob("config/lab/*.yaml")),
        (f"share/{package_name}/config/reference", glob("config/reference/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="adriano",
    maintainer_email="adriano.scibilia@cnr.it",
    description="Config, planning, orchestration, conversion, identification and validation for RR_01.",
    license="Apache-2.0",
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "record_iiwa = erd_recording.pipeline:main_iiwa",
            "record_ur10 = erd_recording.pipeline:main_ur10",
            "show_plan = erd_recording.cli:show_plan_main",
            "erd_link_test = erd_recording.link_test:link_test_main",
            "erd_monitor_test = erd_recording.link_test:monitor_test_main",
            "erd_code_digest = erd_recording.code_digest:main",
        ],
    },
)
