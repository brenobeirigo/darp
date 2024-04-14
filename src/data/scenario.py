from itertools import product
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger()
from pathlib import Path


@dataclass
class ScenarioConfig:
    scenario_id: str
    instance_filepath: str
    is_flex_depot: bool
    max_driving_time_h: float
    allow_rejection: bool
    obj: tuple[str,dict[str,float]]
    cost_per_min: float
    cost_per_km: float
    speed_km_h: float
    revenue_per_load_unit: float
    instance_label: str = ""

    @property
    def obj_label(self):
        return (
            self.obj[0]
            if self.obj[1] is None
            else (
                f"{self.obj[0]}"
                f"(" + ",".join(
                    [
                        f"{v}"
                        for k, v in self.obj[1].items()
                    ]) +")"
            )
        )

    def to_dict(self) -> dict:
        """Convert the ScenarioConfig instance into a dictionary"""
        d = asdict(self)
        d["obj"] = self.obj_label
        return d
    
    def __post_init__(self):

        instance = Path(self.instance_filepath)

        if not instance.exists():
            raise FileNotFoundError(
                f"Instance file path '{self.instance_filepath}' does not exist."
            )

        self.instance_label = str(instance.stem)

    def get_test_label(self, max_time):
        from pathlib import Path

        instance_label = Path(self.instance_filepath).stem
        is_flex_depot = self.is_flex_depot
        obj = self.obj_label
        max_driving_time_h = self.max_driving_time_h
        allow_rejection = self.allow_rejection
        scenario_id = self.scenario_id
        cost_per_min=self.cost_per_min
        cost_per_km=self.cost_per_km
        speed_km_h=self.speed_km_h
        revenue_per_load_unit=self.revenue_per_load_unit

        test_label = "-".join(
            [
                f"scenario_id={scenario_id}",
                f"instance_label={instance_label}",
                f"flex={is_flex_depot}",
                f"obj={obj}",
                f"allow_rejection={allow_rejection}",
                f"max_driving_time_h={max_driving_time_h}",
                f"max_runtime_min={max_time}",
                f"cost_per_min={cost_per_min:.2f}",
                f"cost_per_km={cost_per_km:.2f}",
                f"speed_km_h={speed_km_h}",
                f"revenue_per_load_unit={revenue_per_load_unit}",
            ]
        )
        return test_label


@dataclass
class Scenario:
    scenario_id: str
    description: str
    instances: list[str]
    is_flex_depot: list[bool]
    max_driving_time_h: list[int]
    allow_rejection: list[bool]
    cost_per_min: list[float]
    cost_per_km: list[float]
    speed_km_h: list[float]
    revenue_per_load_unit: list[float]
    obj: list[tuple[str,dict[str,float]]]

    @classmethod
    def read(cls, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        scenarios = {}
        for k, scenario_data in data.items():

            logger.debug(f"Reading scenario '{k}'")

            scenario = cls(
                scenario_id=scenario_data["scenario_id"],
                description=scenario_data["description"],
                instances=scenario_data["instances"],
                is_flex_depot=scenario_data["is_flex_depot"],
                max_driving_time_h=scenario_data["max_driving_time_h"],
                allow_rejection=scenario_data["allow_rejection"],
                cost_per_min=scenario_data["cost_per_min"],
                cost_per_km=scenario_data["cost_per_km"],
                speed_km_h=scenario_data["speed_km_h"],
                obj=[tuple(o_data) for o_data in scenario_data["obj"]],
                revenue_per_load_unit=scenario_data["revenue_per_load_unit"],
            )
            scenarios[k] = scenario

        return scenarios

    def generate_scenario_values(self):

        instances = self.instances
        is_flex_depots = self.is_flex_depot
        max_driving_times = self.max_driving_time_h
        allow_rejections = self.allow_rejection
        objs = self.obj
        cost_per_mins = self.cost_per_min
        cost_per_kms = self.cost_per_km
        speed_km_hs = self.speed_km_h
        revenue_per_load_units = self.revenue_per_load_unit

        for (
            instance_label,
            is_flex_depot,
            max_driving_time_h,
            allow_rejection,
            obj,
            cost_per_min,
            cost_per_km,
            speed_km_h,
            revenue_per_load_unit,
        ) in product(
            instances,
            is_flex_depots,
            max_driving_times,
            allow_rejections,
            objs,
            cost_per_mins,
            cost_per_kms,
            speed_km_hs,
            revenue_per_load_units
        ):
            yield ScenarioConfig(
                scenario_id=self.scenario_id,
                instance_filepath=instance_label,
                is_flex_depot=is_flex_depot,
                max_driving_time_h=max_driving_time_h,
                allow_rejection=allow_rejection,
                obj=obj,
                cost_per_min=cost_per_min,
                cost_per_km=cost_per_km,
                speed_km_h=speed_km_h,
                revenue_per_load_unit=revenue_per_load_unit
            )


if __name__ == "__main__":
    from pprint import pprint
      
    file_path = "C:/Users/AlvesBeirigoB/OneDrive/pkm/PARA/Area/dev/darp/data/raw/mdvrppdtw/scenarios.json"  # Update the file path accordingly
    scenarios = Scenario.read(file_path)

    # Print the list of scenarios
    for k, scenario in scenarios.items():
        print(k)
        pprint(scenario)
