import time

class Timer:

    totals = {}
    starts = {}
    counts = {}

    def start(category):
        Timer.build_category(category)
        Timer.counts[category] += 1
        Timer.starts[category] = time.time()

    def stop(category):
        Timer.totals[category] += time.time() - Timer.starts[category]

    def print():
        for category in Timer.totals:
            print(f"{category}: {Timer.totals[category]}s @ {Timer.totals[category] / Timer.counts[category]}/s - {Timer.counts[category]} times")

    def build_category(category):
        if category not in Timer.totals:
            Timer.totals[category] = 0
        if category not in Timer.starts:
            Timer.starts[category] = 0
        if category not in Timer.counts:
            Timer.counts[category] = 0

# Timer.start('derp')
# time.sleep(0.1)
# Timer.start('dorp')
# time.sleep(0.2)
# Timer.stop('dorp')
# Timer.start('dorp')
# time.sleep(0.1)
# Timer.stop('dorp')
# Timer.stop('derp')
# Timer.print()