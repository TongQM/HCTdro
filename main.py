from classes import *
from findWorstTSPDensity import findWorstTSPDensity

region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 2)
demands = generator.generate()
f = findWorstTSPDensity(region, demands)
# rs, thetas = [d.location.r for d in demands], [d.location.theta for d in demands]
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.scatter(thetas, rs)