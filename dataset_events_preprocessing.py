"""
This script perform a filtering on the events files that will be used for the classification task
based on the methodology described in (LINK ARTICLE).
For early sepsis detection, is for the best interest that the classifier could perform the detections
with a certain period of anticipation, to give time to provide the adequate treatment.
It will remove events that occur in a window prior to the sofa increasing time. The window varies between 4h-8h, at
steps of 2h. Any patient that don't have at least 3h worth of events after the exclusion of the 8h window.
"""