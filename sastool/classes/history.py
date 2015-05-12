import datetime

class SASHistory(object):
    def __init__(self, s=''):
        if isinstance(s, list):
            self._list=s[:]
            return
        if s:
            history_oneliner = s.replace(';;', '<*doublesemicolon*>')
            history_list = [a.strip().replace('<*doublesemicolon*>', ';') \
                            for a in history_oneliner.split(';')]
            history = [a.split(':', 1) for a in history_list]
            self._list = [(float(a[0]), a[1].strip()) for a in history]
        else:
            self._list = []
    def linearize(self):
        history_text = [str(x[0]) + ': ' + x[1] for x in self._list]
        history_text = [a.replace(';', ';;') for a in history_text]
        return '; '.join(history_text)
    def add(self, label, at=None):
        """Add a new entry to the history.

        Inputs:
            label: history text
            at: time of the event. If None, the current time will be used.
        """
        if at is None:
            at = datetime.datetime.now()
        deltat = at - datetime.datetime.fromtimestamp(0, at.tzinfo)
        deltat_seconds = deltat.seconds + deltat.days * 24 * 3600 + deltat.microseconds * 1e-6
        self._list.append((deltat_seconds, label))
    def __str__(self):
        """Return the history in a human-readable format"""
        return '\n'.join([str(h[0]) + ': ' + h[1] for h in self._list])
    def __unicode__(self):
        return '\n'.join([str(h[0]) + ': ' + h[1].encode('utf-8') for h in self._list])
    def pop(self):
        return self._list.pop()
    def __iter__(self):
        return self._list.__iter__()
    def __repr__(self):
        return self.linearize()
