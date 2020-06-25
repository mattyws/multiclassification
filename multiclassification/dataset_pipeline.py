from py._builtin import execfile
import os

# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##            Filtering events from all stays             ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_filter_events.py")
# print("\n############################################################\n")
#
#
# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##           Merging chartevents and labevents            ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_merge_chartevents_labevents.py")
# print("\n############################################################\n")

# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##              Filtering selected features               ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_filter_selected_features.py")
# print("\n############################################################\n")
#
# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##                Merging events by hours                 ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_merge_hourly_events.py")
# print("\n############################################################\n")
#
# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##               Anonymizing medical notes                ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_notes_anonymized_tokens.py")
# print("\n############################################################\n")

# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##              Preprocessing medical notes               ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_notes_preprocessing.py")
# print("\n############################################################\n")

# print("############################################################\n" +
#       "##                                                        ##\n" +
#       "##               Concatenate notes by hours               ##\n" +
#       "##                                                        ##\n" +
#       "############################################################\n")
# execfile("dataset_notes_merge_hourly_events.py")
# print("\n############################################################\n")

print("############################################################\n" +
      "##                                                        ##\n" +
      "##             Creating decompensation dataset            ##\n" +
      "##                                                        ##\n" +
      "############################################################\n")
execfile("decompensation/dataset_create.py")
print("\n############################################################\n")