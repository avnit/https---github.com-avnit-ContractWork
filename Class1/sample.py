print('Hello world')

x = 1
print(x)

x = x+1
print(x)
# Calculate the base rate
rate = input("Enter the rate")
hours = input("no of hours worked")

if float(hours) > 40 :
    pay = (float(rate) * float (hours)) + ((float(hours) - 40 ) * (float(rate) * 1.5))
else :
    pay = float(rate) * float(hours)
# calculate tax
federal = pay * 0.2
state = pay * 0.05
total = pay - federal - state

print("Base Pay you will receive",pay)
print ("federal tax will be" , federal)
print ("state tax will be", state)
print ("Total approx will be ",total)
