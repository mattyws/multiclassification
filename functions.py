from datetime import datetime, timedelta
import time


DATE_PATTERN = "%Y-%m-%d"
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"


def filter_events_before_infection(events, admittime, infection_time, preceding_time,
                                   datetime_pattern=DATETIME_PATTERN, time_key="charttime"):
    """
    Get events that occur from admission time until infection time minus preceding time
    :param events: the events
    :param admittime: the admission time
    :param infection_time: the infection time
    :param preceding_time: the preceding time to get the events
    :param datetime_pattern: the pattern used to store time
    :param key: the dictionary key that has the event time
    :return: 
    """
    admittime_datetime = datetime.strptime(admittime, datetime_pattern)
    infection_datetime = datetime.strptime(infection_time, datetime_pattern) - timedelta(hours=preceding_time)
    new_events = []
    for event in events:
        # Pega a data do evento e o transforma em datetime
        event_datetime = datetime.strptime(event[time_key], datetime_pattern)
        # Compara se o evento aconteceu entre a data de adimissão e a data de infecção (já alterada)
        if event_datetime > admittime_datetime and event_datetime <= infection_datetime:
            new_events.append(event)
    return new_events



def filter_since_time(events_object, time_str, max_interval, datetime_pattern=DATETIME_PATTERN, key="charttime", after=False):
    time_point = time.strptime(time_str, datetime_pattern)
    filtered_objects = []
    for event in events_object:
        if len(event[key]) > 0:
            event_date = time.strptime(event[key], datetime_pattern)
            if after:
                difference = (time.mktime(event_date) - time.mktime(time_point)) / 3600
            else:
                difference = (time.mktime(time_point) - time.mktime(event_date)) / 3600
            if difference >= 0 and difference <= max_interval:
                filtered_objects.append(event)
    return filtered_objects
