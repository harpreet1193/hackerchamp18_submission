from future.builtins import next
import os
import csv
import re
import logging
import optparse
import dedupe
from unidecode import unidecode


#Using Python logging to show verbose output
optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose:
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)


input_file = 'SampleDataset.csv'
output_file = 'csv_example_output.csv'
training_file = 'csv_example_training.json'
settings_file = 'csv_example_learned_settings'
final_file='csv_final_output.csv'

#Data cleaning with the help of Unidecode and Regex.
#Casing, extra spaces, quotes and new lines can be ignored if any present in dataset.
def preProcess(column):
    try : # python 2/3 string differences
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()

    if not column:
        column = None #Missing data is indicated by setting the value to "None"
    return column

#Read data from a CSV file and create a dictionary of records
#The key is a unique record "ID" and each value is dict.
#We have to add a new colomn in our data set as ID if not present already.
def readData(filename):
	global numberOfRecords
	data_d = {}
	with open(filename) as f:
	    reader = csv.DictReader(f)
	    for row in reader:
	        clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
	        row_id = int(row['Id'])
	        data_d[row_id] = dict(clean_row)
	        numberOfRecords+=1
	return data_d

#Comparator to check if two DOB are same or not.
def sameOrNotComparator(field_1, field_2):
	if field_1 and field_2 :
		field_1=str(field_1).split('/')
		field_2=str(field_2).split('/')
		
		if field_1[0] == field_2[0] and field_1[1] == field_2[1] and field_1[2] == field_2[2] :
			return 0
		else:
			return 1


print('importing data ...')
numberOfRecords=0
data_d = readData(input_file)

#Define the fields dedupe will pay attention to
fields = [
		{'field' : 'Id', 'type': 'String'},
        {'field' : 'ln', 'type': 'String'},
        {'field' : 'dob', 'type': 'Custom', 'comparator' : sameOrNotComparator },
        {'field' : 'gn', 'type': 'String'},
        {'field' : 'fn', 'type': 'String',},
        ]

#A new deduper object creation. Pass the Fields to it.
deduper = dedupe.Dedupe(fields)

#Training DeDupe using sample Data.
deduper.sample(data_d, numberOfRecords)

#if trainging data already exists, use it else create on your own.
#if you want to train afresh, delete the previous training file
if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.readTraining(f)


#The least confident records are labeled using users help.
#This helps in creating Training Data set.
print('starting active labeling...')
dedupe.consoleLabel(deduper)

#using the Examples just labeled we traing Dedupe.
#Also learns blocking predicates
deduper.train()


#Save our weights and predicates to disk.
#if the files exist, next time training and learning will be Skipped.
with open(training_file, 'w') as tf:
    deduper.writeTraining(tf)

with open(settings_file, 'wb') as sf:
    deduper.writeSettings(sf)


#Find the threshold that will maximize a weighted average of our precision and recall.
threshold = deduper.threshold(data_d, recall_weight=1)

print('clustering...')

#match return sets of record IDs that dedupe believes are all referring to the same entity.
clustered_dupes = deduper.match(data_d, threshold)

print('# duplicate sets', len(clustered_dupes))

#Write the original data back to a CSV with a new column called 'Cluster ID'
cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id" : cluster_id,
            "canonical representation" : canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1

with open(output_file, 'w') as f_output, open(input_file) as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)

    for row in reader:
        row_id = int(row[0])
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, None)
            row.insert(0, singleton_id)
            singleton_id += 1
            for key in canonical_keys:
                row.append(None)
        writer.writerow(row)

f_output.close()
with open(output_file, 'r') as f_output, open(final_file,'w') as fin_output:
	reader = csv.reader(f_output)
	writer = csv.writer(fin_output)
	head_row=["ID","ln","dob","gn","fn"]
	writer.writerow(head_row)
	uniClusterId=list()
	i=1
	for row in reader:
		if i==1:
			i+=1
			continue
		if row[0] in uniClusterId:
			continue
		else:
			uniClusterId.append(row[0])
			mid_row=(row[2],row[3],row[4],row[5],row[6])
			writer.writerow(mid_row)

