import sys

a = range(5)
print a[4]

# sometimes O(n^2)!
s = ''
for i in range(10000):
    s += str(i)
# O(n)
s = ''
t = []
for i in range(10000):
    t.append(str(i))
s = ''.join(t)

s1 = {'a','b','c','a'} # set, the second 'a' will not be saved
s2 = {'c','d','c','a'} # set, the second 'a' will not be saved
print s1 | s2 # union
print s1 & s2 # intersection
print s1 - s2 # difference

# explicit conversion
y = str(1/9.0)
print y

# try blocks & exceptions
def testTry(n):
    try:
        x = (1.0/n)
    except ZeroDivisionError:
        print('divbyzero')
    else:
        print "result " + str(x)
        pass
    finally:
        print "Done."

print testTry(0)
print testTry(1)

# functions
def square(x):
    """ this is a sample doc string.
        It must be the first thing in the function definition."""
    return x*x

print (square(2))
print square.__doc__

# functions and global variables
# declare global vars with 'global' at the beginning
# of the function or it will be bound to a new local
# variable
badglobal = 3
goodglobal = 3
def writeGlob():
    global goodglobal
    badglobal = 4
    goodglobal = 5
    print badglobal
writeGlob()
print "global %d" % (badglobal,) # still 3!
print "global %d" % (goodglobal,) # now 5

# lambda
square_ = lambda x : x * x
print square_(2)

# generators
def myrange(n):
    i=0
    while i<n:
        yield i
        i += 1
# using it
for j in myrange(4):
    print (j)
# alternative using xrange
# this is lazy!!
for j in xrange(4):
    print (j)

# dictionaries
b = {1: 'a', 'b': 'c'}
print b[1]
print b.items()
# comprehensions with list
print ";".join(["%s=%s" % (k, v) for k, v in b.items()]) # x.split(';') will reverse this

# file example
file_ = open('PySandBox.py','r')
#for line in file_:
#    print(line)
file_.close()

#class definition
class Point:
    commonVar = 1 # shared by all class instances
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        result = Point(0,0) # creates a new Point object
        result.x = self.x + other.x
        result.y = self.y + other.y
        return result

    def __str__(self):
        return 'Point(%f,%f)' % (self.x,self.y)
#using it
p1 = Point(1,2)
p2 = Point(8,9)
print p1
print p1 + p2
# class variables
print p1.commonVar
print p2.commonVar
# you can change the class's commonVar like so:
Point.commonVar = 3
print p1.commonVar # prints 3
print p2.commonVar # prints 3
p1.commonVar = 4 # this changes only p1's commonVar (binds it to a new object)
print p1.commonVar # now 4
print p2.commonVar # is still 3
Point.commonVar = 5
print p2.commonVar # now 5
print p1.commonVar # still 4!

#Inheriting from another class
class PointChild(Point):
    def __init__(self, x):
        self.x = x

pc1 = PointChild("blah")
print pc1.x
print pc1.commonVar # this is 5 as it was for p2
pc1.y = 2 # this adds a y member ONLY to pc1 (not PointChild!)

# this is searched for imports
# you can modify it
print sys.path

# tuples
(a,b,c) = (1,2,3)
print a
print b

# list comprehensions
li = [1,2,3]
print [i*2 for i in li]
# filtering using list comprehensions
print [elem for elem in li if elem < 3]

# introspection
import string
print string.join.__doc__

li1 = ['a','b']
print getattr(li1,"pop") # get a reference to an object (remember that fns are also objects)
getattr(li1,"pop")() # actually call pop using introspection
getattr(li1,"bombo",li1.pop)() # if you try calling a function that doesn't exist,
                               # the third optional argument can be used as a safeguard
print li1

# and
print 'a' and 'b' and 'c' # if all are true, returns last value. prints 'c'
print 'a' and 0   and 'c' # if any is false, returns that value. prints 0

# or
print ''  or 0 or 'c' # if all are false, returns last value. prints 'c'
print 'a' or 0 or 'c' # if any is true, returns that value. prints 'a'

# a?b:c. Useful when you can't use if e.g. in lambdas
a = 0
b = 0
c = 1
(a and [b]) or c # if a is true, becomes [b] or c which returns [b].
                 # Note that 'and' requires ALL to be true to return
                 # the last value. This is why we put b in a list
                 # because even if b itself was false, the list containing
                 # it would still be true (coz non-empty)
                 # if a is false, becomes False or c which returns c
# note that the above gives us either [b] or c. We want either b or c. To do this,
# change c to [c] and add a [0] at the end like so:
((a and [b]) or [c])[0]

# truth table for a?b:c
print "truth table for a?b:c"
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            print str(a) + " " + str(b) + " " + str(c) + " " + str (((a and [b]) or [c])[0])

# equivalently,
print "truth table for a?b:c"
print '\n'.join([str(a) + " " + str(b) + " " + str(c) + " " + str (((a and [b]) or [c])[0])
                 for a in range (2) for b in range (2) for c in range (2)])

# lambdas. cannot have if's. Usually single line
f = (lambda x: x*x)
print f(3)
# or equivalently,
print (lambda x: x*x)(3)

def printallmethods(obj, spacing=15):
    return '\n'.join(["%s %s" % (method.ljust(spacing), (lambda s: " ".join(s.split()))(str(getattr(obj,method).__doc__)))
                      for method in dir(obj) if callable(getattr(obj,method))])
print printallmethods(string,10)
