#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Examp Parabole Mirror Shift"""

import numpy as np
import pkg_resources
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("No instalado")
    import sys
    sys.path.append("../..")


import KrakenOS as Kos

wavelength=0.4

# ______________________________________#

P_Obj = Kos.surf()
P_Obj.Thickness = 25340  # source M1
P_Obj.Diameter = 300  # ???
P_Obj.Drawing = 0 # ???

# ______________________________________#

M1 = Kos.surf()
M1.Rc = -4500000  # -2 * P_Obj.Thickness
M1.Thickness = 0 #M1.Rc / 2
M1.k = 0 #-1.0
M1.Glass = "MIRROR"
M1.Diameter = 500 #300
M1.ShiftY = 0 #500
M1.AxisMove=0

M1.Cylinder_Rxy_Ratio = 1.5
M1.TiltY=90-.26   # 90-0.34  # 6 mrad reflection

# -------------------M2 -------------
M2 = Kos.surf()
M2.Rc = -4500000# -2 * P_Obj.Thickness
M2.Thickness = 0 #M1.Rc / 2
M2.k = 0 #-1.0
M2.Glass = "MIRROR"
M2.Diameter = 500 #300
M2.ShiftY = 0 #500
M2.DespX = -31 #   vertical distance M1 M2
M2.DespZ = 3400 #500  # longitudinal distance M1 M2
M2.AxisMove=0

M2.Cylinder_Rxy_Ratio = 1.5


M2.TiltY=(90-.26) # 90-.26    # 4.5 mrad reflection

# _________________rectangle solid_____M3_________________#

import pyvista as pv

Hy = 0.000001
Lz= 1000.
Wx =500.

p1=np.array([-Wx/2, 0,-Lz/2])
p2=np.array([Wx/2, 0,-Lz/2])
p3=np.array([Wx/2, 0,Lz/2])
p4=np.array([-Wx/2, 0,Lz/2])


DHy=np.array([0,Hy,0])
p5=p1-DHy
p6=p2-DHy
p7=p3-DHy
p8=p4-DHy

vertices = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
faces = np.hstack([[4,0,1,2,3],
                  [4,0,3,7,4],
                  [4,4,5,6,7],
                  [4,5,1,2,6],
                  [4,0,1,5,4],
                  [4,3,2,6,7]]
                  )
Solid = pv.PolyData(vertices, faces)


Prism_SolObj = Kos.surf()

Prism_SolObj.Thickness = 0
Prism_SolObj.Glass = "MIRROR"
Re = [[1,1],[0,0]]
Ab = [[0,0],[1,1]]
# Prism_SolObj.Coating=[Re,Ab,[wavelength,wavelength+.00001],[.26*180/np.pi,np.pi/2-.26*180/np.pi]]

Prism_SolObj.Solid_3d_stl = Solid
Prism_SolObj.Name = 'M3'

Prism_SolObj.TiltX =-.26  # *6 to see something  -.26   4.5 mrad ?
Prism_SolObj.DespY = 0
Prism_SolObj.DespZ = 0
Prism_SolObj.AxisMove = 0

 
# _________________rectangle solid_____M4____________#

Solid2 = pv.PolyData(vertices, faces)


M4 = Kos.surf()

M4.Thickness = 0
M4.Glass = "MIRROR"
# M4.Coating=[Re,Ab,[wavelength,wavelength+0.0001],[.26*180/np.pi,np.pi/2-.26*180/np.pi]]

M4.Solid_3d_stl = Solid2
M4.Name = 'M4'
distM3M4 = 3400

M4.TiltX =-.26  # 
M4.DespY = Hy - M4.TiltX/180*np.pi*2*distM3M4   # -WX to reflect on the right side and -31 to account for reflection laterial shift from M3 
M4.DespZ = distM3M4

M4.AxisMove = 0

import matplotlib.pyplot as plt
from matplotlib.patches import CirclePolygon, RegularPolygon

# ___________M5 ___building a polygon that fits a circular curvature_________ #
# in m
Lengthmirror = 1
RadiusCurvature = 100.
# in mm
Hy = 10
Wx =500.

#---------
Lz = Lengthmirror*1000

def closestodd(nb):
    n= int(nb)
    if n%2==0:
        return n+1
    else:
        return n

coef = closestodd(2*np.pi/(Lengthmirror/RadiusCurvature)) 
nbfacets = coef * 3  # 11  # nb facets on the mirror (ie half circle)
print('nb facets',nbfacets)  # must be odd 
circle = RegularPolygon((0,-RadiusCurvature),2*nbfacets,
                        radius = RadiusCurvature / np.cos(np.pi/2/nbfacets),
                        orientation=np.pi/2, facecolor=None, fill=False)
plt.gca().add_patch(circle)

verts = circle.get_path().vertices
trans = circle.get_patch_transform()
points = trans.transform(verts)
print(points)
vertices2 = np.delete(points, points[:,1]< -RadiusCurvature/4., axis=0)  
# going to 3D
ptsxyz_raw = np.transpose(np.vstack((np.zeros(len(vertices2)),vertices2[:,1],vertices2[:,0])))

x,y,z = ptsxyz_raw.T

cond1 = z>-(Lz+50)/2/1000 #in meter
cond2 = z<(Lz+50)/2/1000 #in meter
#cond3 = y>-100./1000 #in meter

cond = np.where(np.logical_and(cond1, cond2))[0]
ptsxyz=np.take(ptsxyz_raw, cond, axis=0)*1000  # back to millimeter
nbpts = len(ptsxyz)

# sort by increasing z
ptsxyzsorted= np.take(ptsxyz,np.argsort(ptsxyz[:,2]),axis=0)

print('nb pts',len(ptsxyzsorted))
print('ptsxyzsorted',ptsxyzsorted)

# pts in x -Wx/2  lower part of solid
ptwxmin= ptsxyzsorted - np.array([Wx/2,0,0]) # in millimeter
ptwxmax= ptsxyzsorted + np.array([Wx/2,0,0]) # in millimeter


# upper part points of solid
DHy=np.array([0,Hy,0])
pttopwxmin = ptwxmin + DHy
pttopwxmax = ptwxmax + DHy

verticespolygon = np.concatenate((ptwxmin,ptwxmax,pttopwxmin,pttopwxmax))
listfaces = [[4,0,2*nbpts,3*nbpts,nbpts]]  # vert. face  perp z
for ii in range(nbpts-1):
    listfaces.append([4, 0+ii,1+ii,2*nbpts+1+ii,2*nbpts+ii ])  # vert face perp x
    listfaces.append([4, 2*nbpts+ii,2*nbpts+1+ii,3*nbpts+ii+1, 3*nbpts+ii ])  # face perp y  up
    listfaces.append([4, nbpts+ii,3*nbpts+ii, 3*nbpts+1+ii,nbpts+1+ii ])  # vert perp x
    listfaces.append([4, 0+ii,nbpts+ii, nbpts+1+ii,1+ii ])  # horiz face perp y  down
listfaces.append([4,nbpts-1,2*nbpts-1,4*nbpts-1,3*nbpts-1])
print('listfaces',listfaces)
facespolygon = np.hstack(listfaces)
Solid5 = pv.PolyData(verticespolygon, facespolygon)


M5 = Kos.surf()
M5.Thickness = 0
M5.Glass = "MIRROR"
#M5.Glass= "BK7"
M5.Solid_3d_stl = Solid5
M5.Name = 'M5'

M5.TiltX =.26 # -.26  #  -.26   4.5 mrad 
M5.DespY = M4.DespY
M5.DespZ = 5000
M5.AxisMove = 0

# reflectivity
R = [[1.0, 0.0, 1.0],
     [1.0, 0.0, 1.0]]
# absorption
A = [[0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0]]
# wavelength
W = [0.35, 0.4, 0.55]
# angle
THETA = [0, 90]
# anti reflection coating
Solid.Coating =[R, A, W, THETA]

# plt.plot(vertices2[:,0],vertices2[:,1],'o', markersize=10 )
# plt.axis('scaled')
# plt.grid(True)
# plt.show()


# __________________image Plane____________________#

P_Ima = Kos.surf()
P_Ima.Glass = "AIR"
P_Ima.Diameter = 1000
P_Ima.Drawing = 1
P_Ima.Name = "Plano imagen"
P_Ima.Order=0
P_Ima.TiltY=0
P_Ima.DespY = 100 #-1500 #500
P_Ima.DespZ = 6500
P_Ima.AxisMove=0

# _________________elements sequence_____________________#

A = [P_Obj, Prism_SolObj, M4,  M5, P_Ima]
configuracion_1 = Kos.Setup()

# file = ["/home/micha/anaconda3/lib/python3.6/site-packages/KrakenOS/Cat/metalforXrays.AGF"]
# configuracion_1.Load (file)

Espejo = Kos.system(A, configuracion_1)
Rayos = Kos.raykeeper(Espejo)

if 0:
    # ______create rays array coming from ponctual source____(to be corrected)_________________________#
    halfangleopening = 4.5/25340./2.*180/3.14159  * 10  # deg
    tam = 5 # 5   nb rays = tam*2+1
    rad = 100  # # step in x and y
    zdir = rad/np.tan(halfangleopening*np.pi/180.)
    #print('zdir',zdir)
    tsis = len(A) - 1
    for i in range(-tam, tam + 1):
        for j in range(-tam, tam + 1):
            xdir = (i / tam) * rad
            ydir = (j / tam) * rad
            rdir = np.sqrt(xdir**2+ydir**2+zdir**2)
            if np.sqrt(xdir**2+ydir**2) < rad:
                pSource_0 = [0, 0, 0.0]
                dCos = [xdir/rdir, ydir/rdir, zdir/rdir]
                #print('dCos',dCos)
                Espejo.Trace(pSource_0, dCos, wavelength)
                Rayos.push()

if 1:
    # ______create one plane  rays coming from ponctual source_____________________________#

    halfangleopening = 8/25340./2.*180/3.14159    # deg
    halfanglerad = halfangleopening/180*3.14159
    tam = 50  # 5   nb rays = tam*2+1
    for i in range(-tam, tam + 1):
         
        ang = (i / tam) * halfanglerad

        pSource_0 = [0.0, 0.0, 0.0]
        dCos = [0, np.cos(np.pi/2-ang), np.cos(ang)]
        #print('ang',dCos)
        Wl = 0.4
        Espejo.Trace(pSource_0, dCos, Wl)
        Rayos.push()


if 0:
    # ______create parallel rays from rectangular arrays_____________________________#

    tam = 3 # 5   nb rays = tam*2+1
    rad = 150.0   # step in x and y 
    tsis = len(A) - 1
    for i in range(-tam, tam + 1):
        for j in range(-tam, tam + 1):
            x_0 = (i / tam) * rad
            y_0 = (j / tam) * rad
            r = np.sqrt((x_0 * x_0) + (y_0 * y_0))
            if r < rad:
                tet = 0.0
                pSource_0 = [x_0, y_0, 0.0]
                dCos = [0.0, np.sin(np.deg2rad(tet)), np.cos(np.deg2rad(tet))]
                W = 0.004
                Espejo.Trace(pSource_0, dCos, W)
                Rayos.push()

print('intersection pts X Y Z coordinates on all touched optical elements for all rays, source E1, E2...., Efinal')
rayintersects=Rayos.valid_XYZ
print('rays coordinates',rayintersects)

# ___________plots___________________________#

Kos.display2d(Espejo, Rayos, 0)

#Kos.display3d(Espejo, Rayos, 0)

plt.close()
#   plot incoming ray on plane



xyonimage=[]
for rr in rayintersects:
    if len(rr)==len(A): # all elements hit !
        xy=rr[-1][:2]
        xyonimage.append(xy)

ar_xy= np.array(xyonimage)
width = np.amax(ar_xy[:,1])-np.amin(ar_xy[:,1])
fig,ax1=plt.subplots()
ax1.plot(ar_xy[:,0],ar_xy[:,1], 'o', c='b')
ax1.set_title('width  %.2f'%width)

plt.show()

# plt.close()  # beamshape very crude ...
# fig2,ax2=plt.subplots()
# ax2.plot(ar_xy[:,1],np.ones(len(ar_xy)), 'o-', c='b')
# ax2.set_ylim(0,1.1)
# plt.show()


# def R_RMS_delta(Z1, L, M, N, X0, Y0):
#     X1 = ((L / N) * Z1) + X0
#     Y1 = ((M / N) * Z1) + Y0
#     cenX = np.mean(X1)
#     cenY = np.mean(Y1)
#     x1 = (X1 - cenX)
#     y1 = (Y1 - cenY)
#     R2 = ((x1 * x1) + (y1 * y1))
#     R_RMS = np.sqrt(np.mean(R2))
#     return R_RMS

# x,y,z,l,m,n = Rayos.pick(-1, coordinates="local")


# print(R_RMS_delta(z, l, m, n, x, y))