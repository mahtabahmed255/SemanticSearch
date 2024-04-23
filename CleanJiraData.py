import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
import re
import string

def remove_punctuation(tokens):
    return [''.join(c for c in token if c not in string.punctuation) for token in tokens]

def is_proper_english_sentence(text):
    #print("before: ", text)
    # Remove punctuation using translate method
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove punctuation and symbols
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Remove unnecessary spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize the text
    tokens = word_tokenize(text)
    tokens = remove_punctuation(tokens)
    # Check if all tokens are valid English words
    dirtyCount = 0
    for token in tokens:
        if not wordnet.synsets(token):
            dirtyCount += 1
    if len(tokens) != 0:
        if dirtyCount / len(tokens) > 0.35:
            return text, False
    return text, True

with open("/Users/mahtabahmed/output_nokey.txt", "r") as f:
    lines = f.readlines()

final_data = {}
prevKey = None
buffer = []
for line in lines[1:]:
    line = line.rstrip("\n").replace("\\n", "")
    lineSplit = line.split("\t")
    if len(lineSplit) == 2 and line.startswith("SOL"):
        if prevKey is not None:
            final_data[prevKey].extend(buffer)
            buffer = []
        key = lineSplit[0]
        data = lineSplit[1]
        prevKey = key
        if key in final_data:
            final_data[key].append(data)
        else:
            final_data[key] = [data]
    else:
        buffer.append(line)

print("Total Data: ", len(final_data))

'''
i = 0
for key in final_data:
    print(key, final_data[key])
    i+=1
    if i == 100:
        break
'''


i = 0
dirtyData = []
cleanData = []
finalAllData = []
discarded = 0
for key in final_data:
    allData = ' '.join(final_data[key])
    #allData = "jason.whelan@solace.com This feels more like an AD issue than a platform issue..Can your team have a look?.It does look like we may be having some disk access/size issues  We get this log 2018-09-23T02:17:02.437+00:00        watchdog.cpp:1375                       Did not get poll response from StorageElementPollingThread(7)@controlplane(10), sequence number = 101, attempt = 1  2018-09-23T02:17:05.755+00:00        ipciSharedMemObjects.cpp:536            IPC message service time for thread(StorageElementPollingThread(7)@controlplane(10)) msgCode(MSGTYPE_TIMER_EXPIRED) is high(15766858 us)..Threshold(10000000 us) 2018-09-23T02:17:05.756+00:00        watchdog.cpp:861                        Received unexpected poll response from StorageElementPollingThread(7)@controlplane(10) sequence number = 101, attempt = 2  It would be interesting to do a soldebug command on that system that it is possible to easily recreate..If not I'll work with Richard on Wednesday morning to set one up..I have reproduced this on vmr-133-22  {noformat} vmr-133-22(configure/service/msg-backbone)# show message-spool  Config Status:                            Enabled (Primary) Maximum Spool Usage:                      1500 MB Using Internal Disk:                      Yes  Operational Status:                       AD-Active Datapath Status:                          Up Synchronization Status:                   Synced Spool-Sync Status:                        Synced   Last Failure Reason:                    N/A   Last Failure Time:                      N/A Max Message Count:                        100M Message Count Utilization:                100.00% Transaction Resource Utilization:         0.00% Delivered Unacked Msgs Utilization:       0.00% Spool Files Utilization:                  0.00% Active Disk Partition Usage:              38.08% Mate Disk Partition Usage:                -% Next Message Id:                          1 Defragmentation Status:                   Idle Number of delete in-progress:             0                                                ADB         Disk           Total Current Persistent Store Usage (MB)        0.0000       0.0000          0.0000 Number of Messages Currently Spooled            0            0               0 {noformat}  *To reproduce this on vmr-133-22* 1. sting SolOS 8.12.0.1007 and select Max Number of Connections to 100  2. follow steps to manually increase the connection scaling to 200k  {noformat} vmr-133-22(configure)# service msg-backbone shutdown vmr-133-22(configure)# hardware message-spool shutdown vmr-133-22(configure)# system scaling max-connections 200000  [sysadmin@vmr-133-22 ~]$ solacectl service restart [sysadmin@vmr-133-22 ~]$ solacectl cli  vmr-133-22(configure)# no hardware message-spool shutdown vmr-133-22(configure)# no service msg-backbone shutdown {noformat}    This issue also occurs on 100k connection tier..Thanks I was able to reproduce on my VMR,  Currently adding debug code to the load to find out why the pool of free buffer is still stuck with the old value..It seem to only happen when upgrading from very small value ( 100 connections) I could not reproduce when upgrading from 10,000 connections 1..When the system start with a very low Scaling factor this number is set to 100M by default When the number of connection is increased, this number has to be increased to 240M at minimum..Since this does not happen, there is not enough resource left to satisfy both the number of connection and have enough to store messages..Note issue only occurs when upgrading from a very low tier scaling..If so where did it get broken?.You need to do a message spool reset BEFORE enabling it again..This is the read that messed things up  service msg-backbone shutdown hardware message-spool shutdown system scaling max-connections 200000  admin system message-spool reset full  no hardware message-spool shutdown no service msg-backbone shutdown   If the system is in that condition already a new sting will be required   I have a solution tested for this issue..Essentially when reading back the maxMessageCountSchema from the save mirror information, I do not overwrite the previous value if it is found to be bigger..Essentially this allow the better hardware capability to be accepted..Allow the number of max spool messages to be upgrades properly when the number of connection is increased..This is required to have enough ADB blocks to support a high number of connections..Fixed in : build: 100.0dmr_ga.0.46 revision: 146148  Tested and passed on 100.0dmr_ga.0.58 To test, just sting a VMR with enough memory for 100000 connections BUT just string it with 100 connections..This is force the Max Message Count is be 100M (show message-spool detail)..Then service msg-backbone shutdown hardware message-spool shutdown system scaling max-connections 100000  >reload  no hardware message-spool shutdown no service msg-backbone shutdown  which reproduce the problem (note: the max message count is still 100M)..After the fix, the message count is 0.00%  and the max message count was 240M (up from the original 100M)..If the system is in that condition already a new sting will be required  Can you confirm that this statement is true?.Can we not delete some files on the adb directory?.A new instance would need to be created."
    sentences = sent_tokenize(allData)
    keep = []
    for sentence in sentences:
        cleanedText, flag = is_proper_english_sentence(sentence)
        if flag:
            keep.append(cleanedText)
    keep = '.'.join(keep)
    keepTokens = word_tokenize(keep)
    if len(keep) == 0 or len(keepTokens) <10:
        discarded += 1
    else:
        finalAllData.append(keep)
    i += 1
    #if i == 10000:
        #break

print("Kept: ", len(finalAllData))
print("Discarded: ", discarded)
with open("/Users/mahtabahmed/clean_data.txt", "w") as f:
    for data in finalAllData:
        f.write(data + '\n')

'''
# Example usage
text = "I believe this is an addkeyto bug and not a product bug.  I am able to reproduce it. The error is expected the first time addkeyto is run since sshd is restarted, however sshd does not need to restart after.  Roland/Andrew  - What is the next step here? We are still seeing these errors in sanity...is this something we need to whitelist in AFW, or is there still a product/addkeyto bug to be fixed here?  Thanks. It is expected that sshd is restarted the first time addkeyto is run after each reload default config, if it is happening more then that it is unintentional.  If Roland can setup a system where it happens every time, then it will help me debug it.  I have been unable to reproduce it.  After Roland helped get router in this state, we were able to figure out there was a bug in how service sshd restart was done. Change service sshd restart to just stop sshd, since solacedaemon will do the restart  build: 100.0vmr_aws_marketplace.0.9 revision: 124002  Actuall commit, after extra files reverted.  build: 100.0vmr_aws_marketplace.0.9 revision: 124005  I don't think this is fixed.  Here are some results against soltr_100.0vmr_aws_marketplace.0.9 where we still see sshd restarting very frequently.   http://192.168.2.88/sustaining/summary.php?ChildID=1789752 The error first happened before the init logs were collected.  Were can we get the log before this and what was happening to the router? I have reproduced it again. I had the right fix, but it was not applied correctly because the load install has become broken.  I need to find out what other branches the loadinstall could be broken in. Fix service script again remove duplicated code  build: 100.0vmr_aws_marketplace.0.10 revision: 124090  I think I got it fixed this time. Load 10 definitely seems to be much cleaner, no sightings yet. I think we can close this now. Thanks Andrew."
if is_proper_english_sentence(text):
    print("The text is a proper English sentence.")
else:
    print("The text is not a proper English sentence.")
'''