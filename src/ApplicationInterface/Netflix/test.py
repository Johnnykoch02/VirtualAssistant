# Creating this just for testing purposes

from interface import Netflix

n = Netflix()
n.watch("breaking bad")

# Include this to create closing behavior in the terminal
end = False
while not end:
    if input("Press enter to quit") == "":
        end = True
n.quit()