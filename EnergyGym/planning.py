import numpy as np

from datetime import datetime
import time
#from dateutil import tz
from datetime import timezone
from datetime import timedelta
from datetime import tzinfo
import random

"""
timezone
"""
#UTC=tz.gettz('UTC')
UTC=timezone.utc
#CET=tz.gettz('Europe/Paris')
#LOCTZ = tz.gettz('Europe/Paris')
# cf https://docs.python.org/fr/3.6/library/datetime.html#datetime.tzinfo
# chercher datetime.tzinfo dans https://docs.python.org/3/library/datetime.html

ZERO = timedelta(0)
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
#print(ZERO)
#print(SECOND)

STDOFFSET = timedelta(seconds = -time.timezone)
if time.daylight:
    DSTOFFSET = timedelta(seconds = -time.altzone)
else:
    DSTOFFSET = STDOFFSET

DSTDIFF = DSTOFFSET - STDOFFSET


class LocalTimezone(tzinfo):
    """
    A class capturing the platform's idea of local time.
    (May result in wrong values on historical times in
    timezones where UTC offset and/or the DST rules had
    changed in the past.)

    Cf https://docs.python.org/3/library/datetime.html
    """

    def fromutc(self, dt):
        assert dt.tzinfo is self
        stamp = (dt - datetime(1970, 1, 1, tzinfo=self)) // SECOND
        args = time.localtime(stamp)[:6]
        dst_diff = DSTDIFF // SECOND
        # Detect fold
        fold = (args == time.localtime(stamp - dst_diff))
        return datetime(*args, microsecond=dt.microsecond,
                        tzinfo=self, fold=fold)

    def utcoffset(self, dt):
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt):
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt):
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        tt = (dt.year, dt.month, dt.day,
              dt.hour, dt.minute, dt.second,
              dt.weekday(), 0, 0)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0

LOCTZ = LocalTimezone()


"""
a classic week schedule

fixed working hours each day except saturday and sunday

start at 8 and stop at 17
"""
classic=np.array([ [8,17], [8,17], [8,17], [8,17], [8,17], [-1,-1], [-1,-1] ])

def tsToTuple(ts, tz=LOCTZ):
    """
    ts : unix time stamp en s
    tz : timezone as a datutil object

    return date tuple tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst
    """
    _time=datetime.fromtimestamp(ts, tz)
    _tuple=_time.timetuple()
    return(_tuple)

def tsToHuman(ts, fmt="%Y-%m-%d %H:%M:%S:%z", tz=LOCTZ):
    """
    format a timestamp to something readable by a human
    """
    return datetime.fromtimestamp(ts, tz).strftime(fmt)

def inPeriod(d,m,ps,pe):
    """
    ps : period start as "dd-mm"
    pe : period end
    d : day number in the month
    m : month number
    return boolean : True if day-month in period, False otherwise
    """
    inperiod = False
    ps = ps.split("-")
    pe = pe.split("-")
    # si moins d'un mois
    if int(ps[1]) == int(pe[1]):
        if m == int(ps[1]):
            if d >= int(ps[0]) and d <= int(pe[0]):
                inperiod = True
    # si plus d'un mois, 3 cas de figure
    else:
        if m == int(ps[1]) and d >= int(ps[0]):
            inperiod = True
        if m == int(pe[1]) and d <= int(pe[0]):
            inperiod = True
        if m > int(ps[1]) and m < int(pe[1]):
            inperiod = True
    return inperiod

def biosAgenda(nbpts, step, start, offs, schedule=classic):
    """
    un agenda de présence avec prise en compte de jours fériés, voire de périodes de confinement
    ```
    offs =  {
        "2019-offs": [ "01-01", "22-04", "01-05", "08-05", "30-05", "31-05", "10-06", "15-08", "16-08", "01-11", "11-11", ["23-12","25-12"] ],
        "2020-offs": [ "01-01", ["17-03","10-05"], "21-05", "22-05", "01-06", "13-07", "14-07", "11-11", "24-12", "25-12", "31-12"],
        "2021-offs": [ "01-01", "05-04", "13-05", "14-05", "24-05", "14-07", "01-11", "11-11", "12-11", "24-12", "31-12"]
        }
    ```
    """
    time = start
    agenda = np.zeros(nbpts)
    weekend = []
    for i in range(schedule.shape[0]):
        if -1 in schedule[i]:
            weekend.append(i)
    #print(weekend)
    for i in range(0,nbpts):
        tpl = tsToTuple(time)
        y = tpl.tm_year
        d = tpl.tm_mday
        m = tpl.tm_mon
        wd = tpl.tm_wday
        h = tpl.tm_hour
        horaires = schedule[wd]
        # valeur par défaut
        work = 1
        # on applique les jours off s'ils existent
        key = "{}-offs".format(y)
        if key in offs :
          for element in offs[key]:
            if isinstance(element,list):
                if inPeriod(d,m,element[0],element[1]):
                    work = 0
            else:
                off = element.split("-")
                if m == int(off[1]) and d == int(off[0]):
                    work = 0
        # on applique l'agenda hebdo
        if wd in weekend:
            work = 0
        if h not in range(horaires[0], horaires[1]):
            work = 0
        agenda[i] = work
        time+=step
    return agenda

def getLevelDuration(agenda, i):
    """
    return the supposed duration of the level in number of steps

    a level = period during which we can see no change in the agenda
    """
    j=i
    while(agenda[j]==agenda[j+1]):
        if j < agenda.shape[0]-2:
            j+=1
        else:
            break
    return j+1-i

def getRandomStart(start, end, month_min, month_max, year=None):
    """
    tire aléatoirement un timestamp dans un intervalle
    s'assure que le mois du timestamp convient à la saison que l'on veut étudier (hiver, été)
    """
    while True:
        randomts = random.randrange(start, end)
        tpl = tsToTuple(randomts)
        if tpl.tm_mon <= month_max or tpl.tm_mon >=month_min:
            if year is None:
                break
            elif tpl.tm_year == year:
                break
    return randomts
