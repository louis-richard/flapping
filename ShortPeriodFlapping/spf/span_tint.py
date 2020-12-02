
from dateutil import parser as date_parser


def span_tint(axs, tint, **kwargs):
    for axis in axs:
        t_start, t_stop = [date_parser.parse(tint[0]), date_parser.parse(tint[1])]
        axis.axvspan(t_start, t_stop, **kwargs)

    return axs
