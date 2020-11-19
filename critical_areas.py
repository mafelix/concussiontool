from html_utils import ToHtmlList
from enum import Enum

class CritialAreas(Enum):
    PAIN = 'Pain'
    MOBILITY = 'Mobility'
    MOOD = 'Mood'
    COGNITION = 'Cognition'
    SLEEP = 'Sleep'
    DIZZINESS = 'Dizziness'
    FATIGUE = 'Fatigue'
    MEMORY = 'Memory'
    VISUAL_MOTOR_SPEED = 'Visual Motor Speed'
    REACTION_TIME = 'Reaction Time'

    """
        Renders a list of critical areas to an unordered list html.
        input: list of CriticalAreas enums
        outputs: html list
    """
    @staticmethod
    def toHtmlList(critical_areas):
        return ToHtmlList(map(lambda myenum : myenum.value, critical_areas))