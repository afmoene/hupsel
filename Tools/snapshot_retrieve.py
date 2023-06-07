import os
import datetime
import filecmp

def get_snapshots():
    snapshots=os.listdir('/home/user/.snapshots')
    snapshots.sort()
    snapshot_stamp = []
    for snapshot in snapshots:
        mydate = snapshot[:-7]
        mytime = snapshot[11:]
        timestamp = mydate+' '+'%s:%s:%s'%(mytime[:2], mytime[2:4],mytime[4:6])
        timestamp = datetime.datetime.fromisoformat(timestamp)
        snapshot_stamp.append(timestamp)
    # Sort in place
    # snapshot_stamp.sort()
    return (snapshot_stamp, snapshots)

def list_snapshots():
    """
    
    list_snapshots()
    
    Function lists the timestamps of all available snapshots in your Cocalc account.
    Note that the given time is UTC
    
    """
    (snapshot_stamp, snapshots)= get_snapshots()
    for sn in snapshot_stamp:
        print(sn)

def retrieve(fname, around_time, relative=None):
    """
    
    retrieve(fname, around_time, relative=)
    
    Function retrieves a file from a snapshot and copies it to it's original location. Before copying,
    it makes a backup of the file that was currently in that original location. The name of that backup 
    file is appended with the date/time at which the retrieval takes place.
    
    Arguments:
        fname:       the name (including path) of the file to be retrieved, starting in your home directory
                     (e.g. 'hupsel/Step-1/Practical-1.ipynb' )
        around_time: time around which we should search for a snapshot
        relative:    should we retrieve from a snapshot before (relative='before') or after (relative='after')
                     the indicated time
       
    """
    # Which snapshots are available ?
    (snapshot_stamp, snapshots)= get_snapshots()
    
    # Convert the given time limit to a datetime
    try:
        snapshot_time = datetime.datetime.fromisoformat(around_time)
    except:
        print('Error: %s is not a valid format for date/time'%(before_time))
        return
    
    # Search for snapshot that is just newer than the given time limit
    too_old = True
    i = 0
    while (too_old and (i<len(snapshot_stamp))):
        i = i + 1
        if (snapshot_stamp[i] > snapshot_time):
            too_old = False

    if (too_old):
        print('Error: could not find a sufficiently new snapshot')
        return
    
    if (relative == 'before'):
        i = i - 1

    # Check if file exists in the requested snapshot
    if (relative):
        snapname = '/home/user/.snapshots'+'/'+snapshots[i]+'/'+fname
        if (not os.path.isfile(snapname)):
            print('Error: file %s does not exist in snapshot %s'%(fname, snapshots[i]))
            return
    else:
        after_exist = 'File exists in snapshot'
        snapname = '/home/user/.snapshots'+'/'+snapshots[i]+'/'+fname
        if (not os.path.isfile(snapname)):
            after_exist = 'Error: file does not exists in snapshot %s'%(snapshots[i])
        before_exist = 'File exists in snapshot'
        snapname = '/home/user/.snapshots'+'/'+snapshots[i-1]+'/'+fname
        if (not os.path.isfile(snapname)):
            before_exist = 'Error: file does not exist in snapshot %s'%(snapshots[i-1])

        
    if (not relative):
        print('The following snapshots are available around this time:')
        print('%s: %s (%s)'%('before', snapshot_stamp[i-1], before_exist))
        print('%s: %s (%s)'%('after ', snapshot_stamp[i], after_exist))
        print('Choose which snapshot to use, and call the function again, now including the keyword: relative=')
        
    # Check whether the request file exists
    homedir = '/home/user'
    if (not os.path.isfile(homedir+'/'+fname)):
        print('Error: file %s does not exist'%(homedir+'/'+fname))
        return
    else:
        # Check whether files are actually different
        newname = homedir+'/'+fname+'.'+datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S')
        oldname = homedir+'/'+fname
        snapname = '/home/user/.snapshots'+'/'+snapshots[i]+'/'+fname

        if ((not relative==None) and (not filecmp.cmp(oldname, snapname))):
            # Rename original file
            os.system('mv %s %s'%(oldname, newname))
        
            # Copy from snapshot
            os.system('cp %s %s'%(snapname, oldname))
        elif (not relative==None):
            print('Warning: the current file is the same as that from the requested snapshot')


