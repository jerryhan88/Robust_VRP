import os.path as opath
import sys
import pickle

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QSize


# colors = [QColor(255, 0, 0),
#           QColor(50, 55, 100),
#           QColor(0, 0, 255),
#           QColor(171, 121, 66),
#           QColor(148, 32, 146)]


colors = {
    'cyan': "#00ffff", 'darkblue': "#00008b",'darkcyan': "#008b8b",
    'darkmagenta': "#8b008b", 'darkolivegreen': "#556b2f", 'darkorange': "#ff8c00",
    'darkgrey': "#a9a9a9", 'darkgreen': "#006400", 'darkkhaki': "#bdb76b",
    'darkorchid': "#9932cc", 'darkred': "#8b0000", 'darksalmon': "#e9967a",
    'black': "#000000", 'blue': "#0000ff", 'brown': "#a52a2a",
    'aqua': "#00ffff", 'azure': "#f0ffff", 'beige': "#f5f5dc",
    'darkviolet': "#9400d3", 'fuchsia': "#ff00ff", 'gold': "#ffd700",
    'green': "#008000", 'indigo': "#4b0082", 'khaki': "#f0e68c",
    'lightblue': "#add8e6", 'lightcyan': "#e0ffff", 'lightgreen': "#90ee90",
    'lightgrey': "#d3d3d3", 'lightpink': "#ffb6c1", 'lightyellow': "#ffffe0",
    'lime': "#00ff00", 'magenta': "#ff00ff", 'maroon': "#800000",
    'navy': "#000080", 'olive': "#808000", 'orange': "#ffa500",
    'pink': "#ffc0cb", 'purple': "#800080", 'violet': "#800080",
    'red': "#ff0000", 'silver': "#c0c0c0", 'white': "#ffffff",
    'yellow': "#ffff00"
}

pallet = [v for k, v in colors.items() if k != 'black']

fontSize = 20
default_font = QFont('Decorative', fontSize)


mainFrameOrigin = (100, 150)
hMargin, vMargin = 10, 10
xUnit, yUnit = 100, 50
coX, coY = (hMargin * 4, vMargin * 4)


class Node(object):
    def __init__(self, nid, scheduleInputs, drawingInputs):
        self.nid = nid
        self.numTS, self.capa = scheduleInputs
        self.oriX, self.oriY, self.xLen = drawingInputs
        self.tsSchedule = [[None] * self.numTS for _ in range(self.capa)]
        self.demands = []

    def schedule_demand(self, demand, vehicle):
        for j, aPlatSchedule in enumerate(self.tsSchedule):
            for occupyingDemand in aPlatSchedule[demand.s_d : demand.e_d + 1]:
                if occupyingDemand is not None:
                    break
            else:
                for slot in range(demand.s_d, demand.e_d + 1):
                    aPlatSchedule[slot] = demand.did
                demandOriX = self.oriX + demand.s_d * xUnit
                demandOriY = self.oriY + j * yUnit
                demand.initPos(demandOriX, demandOriY)
                self.demands.append(demand)
                #
                routeOriX = self.oriX + demand.a_d * xUnit
                routeOriY = self.oriY + j * yUnit + yUnit / 2
                vehicle.routePos[demand.did] = (routeOriX, routeOriY)
                vehicle.demands[demand.did] = demand
                break

    def draw(self, qp):
        txt = 'n%d' % self.nid
        qp.drawText(self.oriX - hMargin * 3, self.oriY,
                    len(txt) * 15, self.capa * yUnit, Qt.AlignVCenter, txt)

        qp.drawLine(self.oriX, self.oriY, self.oriX + self.xLen, self.oriY)
        for i in range(self.capa):
            yPos = self.oriY + (i + 1) * yUnit
            qp.drawLine(self.oriX, yPos, self.oriX + self.xLen, yPos)

        for demand in self.demands:
            demand.draw(qp)


class Demand(object):
    def __init__(self, did, assigned_vid, s_d, e_d, p_d, a_d, w_d):
        self.did = did
        self.assigned_vid = assigned_vid
        self.s_d, self.e_d, self.p_d, self.a_d, self.w_d = s_d, e_d, p_d, a_d, w_d
        self.xLen = self.p_d * xUnit

    def initPos(self, oriX, oriY):
        self.oriX, self.oriY = oriX, oriY

    def draw(self, qp):
        qp.setBrush(QColor(pallet[self.assigned_vid % len(pallet)]))
        qp.drawRect(self.oriX, self.oriY, self.xLen, yUnit)
        qp.drawText(self.oriX, self.oriY,
                    self.xLen, yUnit, Qt.AlignCenter, 'D%d' % self.did)


class Vehicle(object):
    def __init__(self, vid, route):
        self.vid, self.route = vid, route
        self.vLabel = 'V%d' % self.vid
        self.routePos = {}
        self.demands = {}

    def draw(self, qp):
        pen = QPen(QColor(pallet[self.vid % len(pallet)]), 2.5, Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(Qt.NoBrush)
        lastX, lastY = None, None
        for i, did in enumerate(self.route):
            curX, curY = self.routePos[did]
            if i == 0:
                qp.drawText(curX - xUnit / 3, curY - yUnit / 2,
                            len(self.vLabel) * 15, fontSize * 1.5, Qt.AlignCenter, self.vLabel)
            else:
                qp.drawLine(lastX, lastY, curX, curY)
            lastX = curX + (self.demands[did].w_d + self.demands[did].p_d)  * xUnit
            lastY = curY
            qp.drawLine(curX, curY, lastX, lastY)


class Viz(QWidget):
    def __init__(self, fpath):
        super().__init__()
        #
        inputs, sols = load_input_sol(fpath)
        self.problemName = inputs['problemName']
        H, N, c_i = [inputs.get(k) for k in ['H', 'N', 'c_i']]
        #
        self.numTS = len(H)
        self.xLen = self.numTS * xUnit
        self.yLen = sum(c_i) * yUnit + (len(N) + 1) * vMargin
        #
        self.nodes = []
        for nid in N:
            oriX = coX
            oriY = coY + (nid + 1) * vMargin + sum(c_i[:nid]) * yUnit
            scheduleInputs = (self.numTS, c_i[nid])
            drawingInputs = (oriX, oriY, self.xLen)
            self.nodes.append(Node(nid, scheduleInputs, drawingInputs))
        w, h = coX + self.xLen + hMargin * 2, coY + self.yLen + vMargin * 4
        self.canvasSize = (w, h)
        #
        self.inter_sols(inputs, sols)
        #
        self.image = QImage(w, h, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.clearImage()
        #
        self.initUI()

    def inter_sols(self, inputs, sols):
        n0, V, H, cT = [inputs.get(k) for k in ['n0', 'V', 'H', 'cT']]
        D, Ds, l_d, p_d = [inputs.get(k) for k in ['D', 'Ds', 'l_d', 'p_d']]
        #
        s_d, e_d = [sols.get(k) for k in ['s_d', 'e_d']]
        y_vd, x_hvdd = [sols.get(k) for k in ['y_vd', 'x_hvdd']]
        a_d, w_d = [sols.get(k) for k in ['a_d', 'w_d']]
        #
        self.vehicles = []
        d2v = {}
        for v in V:
            for d in D:
                if y_vd[v, d] > 0.5:
                    d2v[d] = v
            _route = {}
            for h in H:
                for d1 in Ds:
                    for d2 in Ds:
                        if x_hvdd[h, v, d1, d2] > 0.5:
                            _route[d1] = d2
            route = [n0, _route[n0]]
            while route[-1] != n0:
                route.append(_route[route[-1]])
            self.vehicles.append(Vehicle(v, route[1:-1]))

        for d in D:
            demand = Demand(d, d2v[d], int(s_d[d]), int(e_d[d]), int(p_d[d]), a_d[d] / cT, w_d[d] / cT)
            self.nodes[l_d[d]].schedule_demand(demand, self.vehicles[d2v[d]])


    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.white)  ## switch it to else
        self.update()

    def initUI(self):
        w, h = self.canvasSize
        self.setGeometry(mainFrameOrigin[0], mainFrameOrigin[1], w, h)
        self.setWindowTitle('Viz')
        self.setFixedSize(QSize(w, h))
        self.show()

    def paintEvent(self, e):
        for dev in [self, self.image]:
            qp = QPainter()
            qp.begin(dev)
            self.drawCanvas(qp)
            qp.end()

    def save_img(self):
        self.image.save('%s.png' % self.problemName, 'png')

    def drawCanvas(self, qp):
        qp.setFont(default_font)
        qp.drawText(hMargin, vMargin, 200, fontSize * 1.5, Qt.AlignLeft, 'Scenario %s' % self.problemName)
        #
        pen = QPen(Qt.black, 1, Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(Qt.NoBrush)
        qp.drawLine(coX, coY, coX, coY + self.yLen)
        qp.drawLine(coX, coY + self.yLen, coX + self.xLen, coY + self.yLen)
        qp.drawLine(coX + self.xLen, coY + self.yLen,
                    coX + self.xLen - hMargin, coY + self.yLen - vMargin / 2)
        qp.drawLine(coX + self.xLen, coY + self.yLen,
                    coX + self.xLen - hMargin, coY + self.yLen + vMargin / 2)

        pen = QPen(Qt.black, 0.5, Qt.DashLine)
        qp.setPen(pen)
        for ts in range(self.numTS):
            xPos = coX + ts * xUnit
            qp.drawLine(xPos, coY, xPos, coY + self.yLen)
            qp.drawText(xPos, coY + self.yLen + vMargin,
                        xUnit, fontSize * 1.5, Qt.AlignCenter, '%d' % ts)

        for v in self.vehicles:
            v.draw(qp)

        pen = QPen(Qt.black, 1, Qt.SolidLine)
        qp.setPen(pen)
        for n in self.nodes:
            n.draw(qp)




def load_input_sol(fpath):
    with open(fpath, 'rb') as fp:
        inputs, sols = pickle.load(fp)
    return inputs, sols

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fpath = sys.argv[1]
    else:
        fpath = opath.join('_temp', 'is_es0_avg.pkl')
        # fpath = opath.join('_temp', 'is_s1.pkl')
    app = QApplication(sys.argv)
    ex = Viz(fpath)
    ex.save_img()
    sys.exit(app.exec_())