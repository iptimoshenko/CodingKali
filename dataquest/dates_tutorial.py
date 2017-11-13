import csv
with open('guns.csv', 'r') as f:
    csvreader = csv.reader(f)
    data = list(csvreader)
print(data[:5])

import time
current_time = time.time()
current_struct_time = time.gmtime(current_time)
current_year = current_struct_time.tm_year
current_hour = current_struct_time.tm_hour
current_month = current_struct_time.tm_ymon # hr, min, mday (month day)


# Once we have a datetime instance that represents a specific point in time, we can use the following attributes to return more specific properties:
#
# year: returns the year value as an integer.
# month: returns the month value an integer.
# day: returns the day value as an integer.
# hour: returns the hour value as an integer.
# minute: returns the minute value as an integer.
# second: returns the second value as an integer.
# microsecond: returns the microsecond value as an integer.
from datetime import datetime
nye_2017 = datetime(year=2017, month=12, day=31, hour=12, minute=59, second=59)
nye_2017.year
nye_2017.hour
nye_2017.day
nye_2017.second
nye_2017.microsecond
current_time = datetime.now()
current_year = current_time.year
current_month = current_time.month

from datetime import timedelta
today = datetime.now()
diff = timedelta(weeks = 3, days = 2)
future = today + diff
past = today - diff
kirks_birthday = datetime(year=2233, month=3, day=22)
diff = timedelta(weeks = 15)
before_kirk = kirks_birthday - diff

mystery_date = datetime.datetime(year = 2015, month = 12, day = 31, hour=00, minute= 00)
mystery_date_formatted_string= mystery_date.strftime("%I:%M%p on %A %B %d, %Y")
print(mystery_date_formatted_string)

import datetime
mystery_date = datetime.datetime(year = 2003, month = 1, day = 2, hour=00, minute= 00)
mystery_date_formatted_string= mystery_date.strftime("%I:%M%p on %A %B %d, %Y")
mystery_date_2 = datetime.datetime.strptime(mystery_date_formatted_string, "%I:%M%p on %A %B %d, %Y")
print(mystery_date_2)

# convert strings in list of lists:
for post in posts:
    post[2]=datetime.datetime.fromtimestamp(float(post[2]))
march_count = 0
for post in posts:
    march_count += post[2].month == 3