"""tiùe management"""
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from datetime import tzinfo
import time
import random
import numpy as np

# timezone
UTC = timezone.utc
# cf https://docs.python.org/fr/3.6/library/datetime.html#datetime.tzinfo
# chercher datetime.tzinfo dans https://docs.python.org/3/library/datetime.html

ZERO = timedelta(0)
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
# print(ZERO)
# print(SECOND)

STDOFFSET = timedelta(seconds=-time.timezone)
if time.daylight:
    DSTOFFSET = timedelta(seconds=-time.altzone)
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
        return STDOFFSET

    def dst(self, dt):
        if self._isdst(dt):
            return DSTDIFF
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
CLASSIC = np.array([[8, 17], [8, 17], [8, 17], [8, 17], [8, 17], [-1, -1], [-1, -1]])


def tsToTuple(ts, tz=LOCTZ):
    """
    ts : unix time stamp en s
    tz : timezone as a datutil object

    return date tuple tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst
    """
    _time = datetime.fromtimestamp(ts, tz)
    _tuple = _time.timetuple()
    return _tuple


def tsToHuman(ts, fmt="%Y-%m-%d %H:%M:%S:%z", tz=LOCTZ):
    """
    format a timestamp to something readable by a human
    """
    return datetime.fromtimestamp(ts, tz).strftime(fmt)


def in_period(d_m, m_y, p_s, p_e):
    """
    p_s : period start as "dd-mm"
    p_e : period end
    d_m : day number in the month
    m_y : month number in the year
    return boolean : True if day-month in period, False otherwise
    """
    result = False
    p_s = p_s.split("-")
    p_e = p_e.split("-")
    # si moins d'un mois
    if int(p_s[1]) == int(p_e[1]):
        if m_y == int(p_s[1]):
            if int(p_s[0]) <= d_m <= int(p_e[0]):
                result = True
    # si plus d'un mois, 3 cas de figure
    else:
        if m_y == int(p_s[1]) and d_m >= int(p_s[0]):
            result = True
        if m_y == int(p_e[1]) and d_m <= int(p_e[0]):
            result = True
        if int(p_s[1]) < m_y < int(p_e[1]):
            result = True
    return result


def biosAgenda(nbpts, step, start, offs, schedule=CLASSIC):
    """
    un agenda plus abouti avec prise en compte de jours fériés, voire de périodes de confinement
    ```
    offs =  {
        "2019-offs": [ "01-01", "22-04", "01-05", "08-05", "30-05", "31-05", "10-06", "15-08", "16-08", "01-11", "11-11", ["23-12","25-12"] ],
        "2020-offs": [ "01-01", ["17-03","10-05"], "21-05", "22-05", "01-06", "13-07", "14-07", "11-11", "24-12", "25-12", "31-12"],
        "2021-offs": [ "01-01", "05-04", "13-05", "14-05", "24-05", "14-07", "01-11", "11-11", "12-11", "24-12", "31-12"]
        }
    ```
    """
    _time = start
    agenda = np.zeros(nbpts)
    weekend = []
    for i in range(schedule.shape[0]):
        if -1 in schedule[i]:
            weekend.append(i)
    #print(weekend)
    for i in range(0, nbpts):
        tpl = tsToTuple(_time)
        year = tpl.tm_year
        d_m = tpl.tm_mday
        m_y = tpl.tm_mon
        d_w = tpl.tm_wday
        hour = tpl.tm_hour
        horaires = schedule[d_w]
        # valeur par défaut
        work = 1
        # on applique les jours off s'ils existent
        key = f'{year}-offs'
        if key in offs :
            for element in offs[key]:
                if isinstance(element, list):
                    if in_period(d_m, m_y, element[0], element[1]):
                        work = 0
                else:
                    off = element.split("-")
                    if m_y == int(off[1]) and d_m == int(off[0]):
                        work = 0
        # on applique l'agenda hebdo
        if d_w in weekend:
            work = 0
        if hour not in range(horaires[0], horaires[1]):
            work = 0
        agenda[i] = work
        _time += step
    return agenda


def get_level_duration(agenda, i):
    """
    return the supposed duration of the level in number of steps

    a level = period during which we can see no change in the agenda
    """
    j = i
    while agenda[j] == agenda[j+1]:
        if j < agenda.shape[0]-2:
            j += 1
        else:
            break
    return j + 1 - i


def get_random_start(start, end, month_min, month_max, year=None):
    """
    tire aléatoirement un timestamp dans un intervalle
    s'assure que le mois du timestamp convient à la saison que l'on veut étudier (hiver, été)
    """
    while True:
        randomts = random.randrange(start, end)
        tpl = tsToTuple(randomts)
        if tpl.tm_mon <= month_max or tpl.tm_mon >= month_min:
            if year is None:
                break
            if tpl.tm_year == year:
                break
    return randomts
