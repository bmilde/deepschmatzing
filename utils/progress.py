def prog_iterate(it, verbosity=100, key=repr, estimate=None, persec=True):
    """An iterator that yields everything from `it', but prints progress
       information along the way, including time-estimates if
       possible"""
    from itertools import islice
    from datetime import datetime
    import sys

    now = start = datetime.now()
    elapsed = start - start

    # try to guess at the estimate if we can
    if estimate is None:
        try:
            estimate = len(it)
        except:
            pass

    def timedelta_to_seconds(td):
        return td.days * (24*60*60) + td.seconds + (float(td.microseconds) / 1000000)
    def format_timedelta(td, sep=''):
        ret = []
        s = timedelta_to_seconds(td)
        if s < 0:
            neg = True
            s *= -1
        else:
            neg = False

        if s >= (24*60*60):
            days = int(s//(24*60*60))
            ret.append('%dd' % days)
            s -= days*(24*60*60)
        if s >= 60*60:
            hours = int(s//(60*60))
            ret.append('%dh' % hours)
            s -= hours*(60*60)
        if s >= 60:
            minutes = int(s//60)
            ret.append('%dm' % minutes)
            s -= minutes*60
        if s >= 1:
            seconds = int(s)
            ret.append('%ds' % seconds)
            s -= seconds

        if not ret:
            return '0s'

        return ('-' if neg else '') + sep.join(ret)
    def format_datetime(dt, show_date=False):
        if show_date:
            return dt.strftime('%Y-%m-%d %H:%M')
        else:
            return dt.strftime('%H:%M:%S')
    def deq(dt1, dt2):
        "Indicates whether the two datetimes' dates describe the same (day,month,year)"
        d1, d2 = dt1.date(), dt2.date()
        return (    d1.day   == d2.day
                and d1.month == d2.month
                and d1.year  == d2.year)

    sys.stderr.write('Starting at %s\n' % (start,))

    # we're going to islice it so we need to start an iterator
    it = iter(it)

    seen = 0
    while True:
        this_chunk = 0
        thischunk_started = datetime.now()

        # the simple bit: just iterate and yield
        for item in islice(it, verbosity):
            this_chunk += 1
            seen += 1
            yield item

        if this_chunk < verbosity:
            # we're done, the iterator is empty
            break

        now = datetime.now()
        elapsed = now - start
        thischunk_seconds = timedelta_to_seconds(now - thischunk_started)

        if estimate:
            # the estimate is based on the total number of items that
            # we've processed in the total amount of time that's
            # passed, so it should smooth over momentary spikes in
            # speed (but will take a while to adjust to long-term
            # changes in speed)
            remaining = ((elapsed/seen)*estimate)-elapsed
            completion = now + remaining
            count_str = ('%d/%d %.2f%%'
                         % (seen, estimate, float(seen)/estimate*100))
            completion_str = format_datetime(completion, not deq(completion,now))
            estimate_str = (' (%s remaining; completion %s)'
                            % (format_timedelta(remaining),
                               completion_str))
        else:
            count_str = '%d' % seen
            estimate_str = ''

        if key:
            key_str = ': %s' % key(item)
        else:
            key_str = ''

        # unlike the estimate, the persec count is the number per
        # second for *this* batch only, without smoothing
        if persec and thischunk_seconds > 0:
            persec_str = ' (%.1f/s)' % (float(this_chunk)/thischunk_seconds,)
        else:
            persec_str = ''

        sys.stderr.write('%s%s, %s%s%s\n'
                         % (count_str, persec_str,
                            format_timedelta(elapsed), estimate_str, key_str))

    now = datetime.now()
    elapsed = now - start
    elapsed_seconds = timedelta_to_seconds(elapsed)
    if persec and seen > 0 and elapsed_seconds > 0:
        persec_str = ' (@%.1f/sec)' % (float(seen)/elapsed_seconds)
    else:
        persec_str = ''
    sys.stderr.write('Processed %d%s items in %s..%s (%s)\n'
                     % (seen,
                        persec_str,
                        format_datetime(start, not deq(start, now)),
                        format_datetime(now, not deq(start, now)),
                        format_timedelta(elapsed)))
