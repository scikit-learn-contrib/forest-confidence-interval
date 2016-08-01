# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### #
"""

Due-credit
==========

`duecredit <http://duecredit.org>`_  is a framework conceived to address the
problem of inadequate citation of scientific software and methods. It automates
the insertion of citations into code. We use it here to refer to the original
publication introducing the method we have implemented.

See  https://github.com/duecredit/duecredit/blob/master/README.md for examples.

Origin:     Originally a part of the duecredit software package

Copyright:  2015-2016  DueCredit developers

License:    BSD-2
"""

__version__ = '0.0.5'


class _InactiveDueCreditCollector(object):
    """Just a stub at the Collector which would not do anything"""
    def _donothing(self, *args, **kwargs):
        """Perform no good and no bad"""
        pass

    def dcite(self, *args, **kwargs):
        """If I could cite I would"""
        def nondecorating_decorator(func):
            return func
        return nondecorating_decorator

    cite = load = add = _donothing

    def __repr__(self):
        return self.__class__.__name__ + '()'


def _donothing_func(*args, **kwargs):
    """Perform no good and no bad"""
    pass

try:
    from duecredit import due as _due
    from duecredit import BibTeX as _BibTeX
    from duecredit import Doi as _Doi
    from duecredit import Url as _Url
    if '_due' in locals() and not hasattr(_due, 'cite'):
        raise RuntimeError(
            "Imported due lacks .cite. DueCredit is now disabled")
except Exception as e:
    if type(e).__name__ != 'ImportError':
        import logging
        logging.getLogger("duecredit").error(
            "Failed to import duecredit due to %s" % str(e))
    # Initiate due stub
    _due = _InactiveDueCreditCollector()
    _BibTeX = _Doi = _Url = _donothing_func

# Emacs mode definitions
# Local Variables:
# mode: python
# py-indent-offset: 4
# tab-width: 4
# indent-tabs-mode: nil
# End:
