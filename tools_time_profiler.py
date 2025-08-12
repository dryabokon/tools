import time
import pandas as pd
from contextlib import contextmanager
# --------------------------------------------------------------------------
import tools_DF
# --------------------------------------------------------------------------
class Time_Profiler:
    def __init__(self, verbose=True):
        self.current_event = None
        self.current_start = {}  # {event: start_perf_counter}
        self.dict_event_time = {}  # {event: total_seconds}
        self.dict_event_cnt  = {}  # {event: finished_intervals}
        self.verbose = verbose

    # --------------------------------------------------------------------------
    def tic(self, event, reset=False, verbose=None):

        now = time.perf_counter()
        verbose = self.verbose if verbose is None else verbose

        if reset:
            self.dict_event_time.pop(event, None)
            self.dict_event_cnt.pop(event, None)
            if self.current_event == event:
                self.current_event = None
                self.current_start.pop(event, None)

        if self.current_event is not None:
            prev = self.current_event
            start = self.current_start.pop(prev, None)
            if start is not None:
                delta = now - start
                self.dict_event_time[prev] = self.dict_event_time.get(prev, 0.0) + delta
                self.dict_event_cnt[prev]  = self.dict_event_cnt.get(prev, 0) + 1
                if verbose:
                    print("stop  -", prev, f"(+{delta:.3f}s)")
            # If toggling the same event, we just stopped it and we’re done
            if prev == event:
                self.current_event = None
                return


        self.current_event = event
        self.current_start[event] = now

        self.dict_event_time.setdefault(event, 0.0)
        self.dict_event_cnt.setdefault(event, 0)
        if verbose:
            print("start -", event)

    # --------------------------------------------------------------------------
    def stop(self, verbose=None):

        if self.current_event is None:
            return
        self.tic(self.current_event, verbose=verbose)

    # --------------------------------------------------------------------------
    def print_duration(self, event, verbose=None):

        verbose = self.verbose if verbose is None else verbose
        total = self.get_duration_sec(event)
        if total is None:
            return None

        if total < 60:

            value = f"{total:0.3f}s"
        elif total < 3600:
            value = time.strftime('%M:%S', time.gmtime(total))
        else:
            value = time.strftime('%H:%M:%S', time.gmtime(total))

        if verbose:
            print(value, '-', event)
        return value

    # --------------------------------------------------------------------------
    def get_duration_sec(self, event):

        if (event not in self.dict_event_time) and (event not in self.current_start):
            return None
        total = self.dict_event_time.get(event, 0.0)
        if self.current_event == event and event in self.current_start:
            total += time.perf_counter() - self.current_start[event]
        return total

    # --------------------------------------------------------------------------
    def to_dataframe(self):


        events = set(self.dict_event_time.keys()) | set(self.dict_event_cnt.keys()) | set(self.current_start.keys())

        rows = []
        for e in sorted(events):
            time_total = self.get_duration_sec(e) or 0.0
            count = self.dict_event_cnt.get(e, 0)
            time_avg = (time_total / count) if count > 0 else None
            fps = (1.0 / time_avg) if (time_avg and time_avg > 0) else None
            rows.append({
                'event': e,
                'count': count,
                'time_total': time_total,
                'time_avg': time_avg,
                'fps': fps
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=['time_avg', 'time_total'], ascending=[False, False], na_position='last')
        return df.reset_index(drop=True)

    # --------------------------------------------------------------------------
    def stage_stats(self, filename_out):

        df = self.to_dataframe()

        if df.empty:
            with open(filename_out, 'w') as f:
                f.write("No stats recorded.\n")
            return

        total_time = df['time_total'].sum()
        total_count = df['count'].max()
        total_fps = total_count/ (total_time + 1e-6)
        total_avg = total_time / (total_count + 1e-6)

        total_row = pd.DataFrame([{
            'event': 'TOTAL',
            'count': total_count,
            'time_total': total_time,
            'time_avg': total_avg,
            'fps': total_fps
        }])

        out = pd.concat([df, total_row], ignore_index=True)

        try:


            res = tools_DF.prettify(out, showindex=False)
            text = res
        except Exception:
            text = out.to_string(index=False)

        with open(filename_out, 'w') as f:
            f.write(text)

    # --------------------------------------------------------------------------
    def print_stats(self):
        df = self.to_dataframe()
        if df.empty:
            print("No stats recorded.")
            return
        for _, r in df.iterrows():
            avg = f"{r['time_avg']:.6f}s" if pd.notnull(r['time_avg']) else "-"
            fps = f"{r['fps']:.2f}" if pd.notnull(r['fps']) else "-"
            print(f"{avg}\t{fps} FPS\t{r['event']}")

    # --------------------------------------------------------------------------
    def prettify(self, value):
        if value >= 1e9:   return f'{value / 1e9:.2f}G'
        if value >= 1e6:   return f'{value / 1e6:.2f}M'
        if value >= 1e3:   return f'{value / 1e3:.2f}k'
        return f'{value:.2f}'

    # --------------------------------------------------------------------------
    @contextmanager
    def timer(self, event, verbose=None):
        self.tic(event, verbose=verbose)
        try:
            yield
        finally:
            self.stop(verbose=verbose)

    # --------------------------------------------------------------------------
    # Demo helpers
    def funA(self): time.sleep(1.2)
    def funB(self): time.sleep(0.3)

    def test(self):
        for _ in range(10):
            self.tic('A'); self.funA(); self.tic('A')  # toggle A
            self.tic('B'); self.funB(); self.tic('B')  # toggle B
        self.print_stats()
    # --------------------------------------------------------------------------