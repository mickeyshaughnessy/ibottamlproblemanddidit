def inner_median(x,y):
    # returns the median of the intersection of the lists x and y
    inter = sorted(list(set([a for a in x if a in y])))
    _len = len(inter)
    if _len == 0: raise ValueError("There were no common elements in x, y")
    if _len % 2 == 1: 
        return inter[(_len - 1)/2] # return the middle element
    return 0.5*(inter[(_len)/2] + inter[(_len)/2 - 1]) # average of the two middle-most elements

if __name__ == "__main__":
    print inner_median([1], [1]), ' should be 1'
    print inner_median([1,1,3,2,1,3], [1,2,3,4]), ' should be 2'
    print inner_median([3,1,2], [1,2,3,4]), ' should be 2'
    print inner_median([1,3,2,1,3], [1,2,3,4]), ' should be 2'
    print 'should be an error next...'
    print inner_median([],[])
