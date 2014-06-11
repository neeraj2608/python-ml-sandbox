# Getting up to speed with Python
import sys

a = range(5)
assert a[4] == 4

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
assert s1 | s2 == {'a','b','c','d'} # union
assert s1 & s2 == {'a','c'} # intersection
assert s1 - s2 == {'b'} # difference

# explicit conversion
y = str(1/9.0)
assert y == '0.111111111111'

# try blocks & exceptions
def testTry(n):
    try:
        x = (1.0/n)
    except ZeroDivisionError:
        #print 'divbyzero'
        pass
    else:
        return "result " + str(x)
    finally:
        return "done"

assert testTry(0) == 'done' # finally is always executed
assert testTry(1) == 'done' # finally is always executed

# functions
def square(x):
    """ this is a sample doc string.
        It must be the first thing in the function definition."""
    return x*x

assert (square(2)) == 4
#print square.__doc__

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

assert goodglobal == 3
writeGlob()
assert badglobal == 3
assert goodglobal == 5

# lambda
square_ = lambda x : x * x
assert square_(2) == 4

# generators
def myrange(n):
    i=0
    while i<n:
        yield i
        i += 1
# using it
for j in myrange(4):
    #print (j)
    pass
# alternative using xrange
# this is lazy!!
for j in xrange(4):
    #print (j)
    pass

# dictionaries
b = {1: 'a', 'b': 'c'}
assert b[1] == 'a'
assert b.items() == [(1, 'a'), ('b', 'c')]
# comprehensions with list
assert ";".join(["%s=%s" % (k, v) for k, v in b.items()]) == '1=a;b=c' # x.split(';') will reverse this

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
assert (p1 + p2).x == 9 # uses the __add__ definition
assert (p1 + p2).y == 11 # uses the __add__ definition
# class variables
assert p1.commonVar == 1
assert p2.commonVar == 1
# you can change the class's commonVar like so:
Point.commonVar = 3
assert p1.commonVar == 3 # prints 3
assert p2.commonVar == 3 # prints 3
p1.commonVar = 4 # this changes only p1's commonVar (binds it to a new object)
assert p1.commonVar == 4 # now 4
assert p2.commonVar == 3 # is still 3
Point.commonVar = 5
assert p2.commonVar == 5# now 5
assert p1.commonVar == 4 # still 4!

#Inheriting from another class
class PointChild(Point):
    def __init__(self, x):
        self.x = x

pc1 = PointChild("blah")
assert pc1.x == 'blah'
assert pc1.commonVar == 5 # this is 5 as it was for p2
pc1.y = 2 # this adds a y member ONLY to pc1 (not PointChild!)

# this is searched for imports
# you can modify it
#print sys.path

# tuples
(a,b,c) = (1,2,3)
assert a == 1
assert b == 2

# list comprehensions
li = [1,2,3]
assert [i*2 for i in li] == [2,4,6]
# filtering using list comprehensions
assert [elem for elem in li if elem < 3] == [1,2]

# introspection
#import string
#print string.join.__doc__

li1 = ['a','b']
assert hasattr(li1,"pop") # get a reference to an object (remember that fns are also objects)
getattr(li1,"pop")() # actually call pop using introspection
getattr(li1,"bombo",li1.pop)() # if you try calling a function that doesn't exist,
                               # the third optional argument can be used as a safeguard
assert not li1

# and
assert ('a' and 'b' and 'c') == 'c' # if all are true, returns last value. prints 'c'
assert ('a' and 0   and 'c') == 0 # if any is false, returns that value. prints 0

# or
assert (''  or 0 or 'c') == 'c' # if all are false, returns last value. prints 'c'
assert ('a' or 0 or 'c') == 'a' # if any is true, returns that value. prints 'a'

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
#print "truth table for a?b:c"
#for a in [0,1]:
#    for b in [0,1]:
#        for c in [0,1]:
#            print str(a) + " " + str(b) + " " + str(c) + " " + str (((a and [b]) or [c])[0])

# equivalently,
#print "truth table for a?b:c"
#print '\n'.join([str(a) + " " + str(b) + " " + str(c) + " " + str (((a and [b]) or [c])[0])
#                 for a in range (2) for b in range (2) for c in range (2)])

# lambdas. cannot have if's. Usually single line
f = (lambda x: x*x)
assert f(3) == 9
# or equivalently,
assert (lambda x: x*x)(3) == 9

def printallmethods(obj, spacing=15):
    return '\n'.join(["%s %s" % (method.ljust(spacing), (lambda s: " ".join(s.split()))(str(getattr(obj,method).__doc__)))
                      for method in dir(obj) if callable(getattr(obj,method))])
#print printallmethods(string,10)

# Generators
# Compare this:
assert sum([x*x for x in range(0,10)]) == 285
# with this:
assert sum(x*x for x in xrange(0,10)) == 285
# the second one is LAZY. It only computes values as they are needed. It also doesn't compute intermediate values. So, whereas the first
# code will generate the entire list BEFORE evaluating the sum, the second will just calculate the individual items and sum them as it
# goes along.
# xrange is also lazy
# enumerate is also lazy

# Here's another use of generators
def my_range(stop):
    val = 0
    while val < stop:
        yield val
        val += 1
# here, yield makes my_range return a generator object
# observe:
import types
assert type(my_range(1)) is types.GeneratorType
# the generator object is an iterator. e.g. it has a next method
# observe:
assert hasattr(my_range(1),'next')
# exiting the iteration can be done by raising a StopIteration
# or by falling off the end of the generator code.
# the iterator is what's used by the for loop below:
assert [i for i in my_range(3)] == [0,1,2]

# Classmethods
# this java code:
# public class X{
#    public static String a = "hello"
#    public static void printA(){
#        System.out.println(X.a);
#    }
#  }
# is equivalent to
a = "Hello"
def printA():
    print a
# it is NOT equivalent to
class X:
    a = "hello"
    def printA(cls):
        print X.a
    printA = classmethod(printA) # note that @classmethod is a decorator alternative to doing this
# so why have a classmethod builtin at all?
# classmethods are different in that they are IMPLICITLY
# passed in the class object they were invoked
# on as the first argument.
# observe below:
class X:
    a = None
    def printA(cls, a="hello"):
        cls.a = a
        return cls.__name__+".a = "+cls.a
    printA = classmethod(printA)

class X_Child1(X):
    a = None

# note that in the calls below, we don't pass in the cls argument. It is passed
# in implicitly
assert X().printA() == 'X.a = hello' # prints "hello". Sets the 'a' of X
assert X_Child1().printA("hi") == 'X_Child1.a = hi' # prints "hi". Sets the 'a' of X_Child1
# what happens here is that when printA is invoked on
# X_Child1, the a being accessed is the a belonging to
# the X_Child CLASS, and not the X_Child() instance object.
# To illustrate that instances have nothing to do with this,
# this also works:
assert X.printA() == 'X.a = hello' # Compare with above. 'X' instead of 'X()'. still prints "hello"
assert X_Child1.printA("hi") == 'X_Child1.a = hi' # Compare with above. 'X_Child1' instead of 'X_Child1()'. still prints "hi"
assert X_Child1.printA() == 'X_Child1.a = hello' # Note that this will return hello which is the default value of X.printA's a argument

# Here's another feature of classmethods:
# subclasses can redefine the behavior of classmethods of their parents
# Observe:
class X_Child2(X):
    a = None
    def printA(cls, a):
        cls.a = a
        return cls.__name__+".a*2 = "+cls.a*2
    printA = classmethod(printA)

assert X().printA() == 'X.a = hello' # still prints "hello"
assert X_Child2.printA("hi") == 'X_Child2.a*2 = hihi' # now prints "hihi"
# doing this is NOT possible in Java because subclasses CANNOT override STATIC
# methods of their superclasses (they can override instance methods but that's
# another story)

# Here's something else to be aware of if you're
# trying to invoke the super class's classmethod
class X(object):
    a = None
    @classmethod
    def printA(cls, a="hello"):
        cls.a = a
        return cls.__name__+".a = "+cls.a

class X_Child3(X):
    a = None
    @classmethod
    def printA(cls, a):
        cls.a = a
        return (cls.__name__+".a*2 = "+cls.a*2,
        #X.printA(cls, a) # this will throw an error because when we invoke X's printA, it is already getting passed
        #                 # in a cls class object as the first argument
        super(X_Child3, cls).printA("hi"), # NOTE: this will work only if the superclass X extends from object i.e.,
                                           # class X(object): ...
                                           # this is a limitation imposed by super
        super(X_Child3, cls).printA()) # NOTE: this will work only if the superclass X extends from object i.e.,
                                       # this is a limitation imposed by super

assert X.printA() == 'X.a = hello' # still prints "hello"
assert X_Child3.printA("hi") == ('X_Child3.a*2 = hihi','X_Child3.a = hi','X_Child3.a = hello') # prints "hihi" then "hi" from X's printA then "hello" from X's printA
                                                                                               # note that cls is the class object that the method was invoked upon (in
                                                                                               # this case, X_Child3 and NOT X)

# Properties (aka fuck-you Java getters and setters)
# Let's start here:
class X:
    def __init__(self, email=None):
        self.email = email
x = X()
x.email = "blahblah"
assert x.email == 'blahblah' # works as expected
# Now let's add some validation on the email
class X:
    def __init__(self, email=None):
        self.email = email

    def setEmail(self, email=None):
        if email == None or not ('@' in email):
            pass
        else:
            self.email = email

    def getEmail(self):
        return self.email

    email = property(getEmail, setEmail)

x = X()
x.setEmail("blahblah")
assert x.getEmail() == None # prints None as expected
x.setEmail("blah@blah.com")
assert x.getEmail() == 'blah@blah.com' # prints 'blah@blah.com' as expected
# this is the beautiful part. The old way of accessing the instance variable
# directly still works! Observe:
x.email = 'blah1@blah.com'
assert x.email == 'blah1@blah.com' # still works! prints 'blah1@blah.com'
# a. No need to make the variable private (there's no such thing in Python anyway).
# b. Old client code doesn't need to change! Compare that with Java ugliness.
# c. Setter/Getter logic can be added at any time.

# Mix-ins
# Think of mixins as interfaces that are already implemented. Mixins let you add
# small pieces of functionality to your class.
