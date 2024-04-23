from gym import ObservationWrapper
from pandas import DataFrame
from codetiming import Timer
class DetectObjects(ObservationWrapper):


    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector


    def observation(self, observation) -> DataFrame:
        positions = self.detector.detect(observation)

        return positions
