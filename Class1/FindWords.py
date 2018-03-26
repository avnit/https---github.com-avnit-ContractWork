handle = open("File.csv","r")
foreach line in handle :
    if line.match("/d+") :
        print('match found',line)
