import csv
with open('guns.csv', 'r') as f:
    csvreader = csv.reader(f)
    data = list(csvreader)
print(data[:5])

# We can also define global variables inside local scopes:

def test_function():
    global a
    a = 10

test_function()
print(a)

# When we use a variable anywhere in a Python script, the Python interpreter will look for its value according to some simple rules. It will:
# Start with the local scope, if any. If the variable is defined here, it will use that value.
    # Look at any enclosing scopes, starting with the innermost. These are "outside" local scopes. If the variable is defined in any of them, it will use the value.
    # Look in the global scope. If the variable is there, it uses the value.
    # Look in the built-in functions.
    # Throw an error if it doesn't find the variable.
    # A simple way to remember this is LEGBE, which stands for "Local, Enclosing, Global, Built-ins, Error".