"""
2-dimensional euler equations
"""

__author__ = 'balshark(Twitter: @balsharkPhD)'
__version__ = '1.0.0'
__date__ = '03/21/2022'
__status__ = 'Development'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# print("   initialize...",end="")
nx0 = 100       # grid sise
xg = 1          # ghost grid

nx = nx0 + xg*2 # total grid size
ny = 100
nt = 300        # total time step
dx = 10**(-2)   # size between grids
dy = 10**(-2)
dt = 10**(-3)   # time step size

gam = 1.4       # specific heat ratio
bet = gam - 1.0

x = np.empty((nx,ny))
y = np.empty((nx,ny))

ql = np.empty(4)
qr = np.empty(4)

qf = np.empty((nx,ny,4))
qc = np.empty((nx,ny,4))
fl = np.empty((nx+1,ny+1,4))
rhs = np.empty((nx,ny,4))

fig = plt.figure(figsize=plt.figaspect(0.33))
ax1 = fig.add_subplot(131,projection='3d')
ax2 = fig.add_subplot(132,projection='3d')
ax3 = fig.add_subplot(133,projection='3d')
ax1.set_title('r [kg/m^3]')
ax2.set_title('u [m/s]')
ax3.set_title('p [Pa]')
ax1.auto_scale_xyz([0,1],[0,1],[0,1])
ax2.auto_scale_xyz([0,1],[0,1],[0,1])
ax3.auto_scale_xyz([0,1],[0,1],[0,1])
ims = []

# print("...done")

class Euler():
    
    def main(self):
        print('start 2-D Euler simulation')
        self.setup()

        self.calmain()

        self.imsave() 
        
    def setup(self):
        global ql,qr,qf,qc,x,y
        print('   setup...',end='')
        r = np.empty((nx,ny))
        u = np.empty((nx,ny))
        v = np.empty((nx,ny))
        p = np.empty((nx,ny))
        e = np.empty((nx,ny))

        for i in range(nx):
            for j in range(ny):
                u[i][j] = 0.0
                v[i][j] = 0.0
                if i <= nx/2:
                    r[i][j] = 1.0
                    p[i][j] = 1.0
                else:
                    r[i][j] = 0.125
                    p[i][j] = 0.1
                
                e[i][j] = p[i][j]/bet \
                        + 0.5*r[i][j]*(u[i][j]**2+v[i][j]**2)
            
        ql[0] = r[0][0]
        ql[1] = r[0][0]*u[0][0]
        ql[2] = r[0][0]*v[0][0]
        ql[3] = e[0][0]
        qr[0] = r[nx-1][0]
        qr[1] = r[nx-1][0]*u[nx-1][0]
        qr[2] = r[nx-1][0]*v[nx-1][0]
        qr[3] = e[nx-1][0]

        for i in range(nx):
            for j in range(ny):
                x[i][j] = i*dx-dx/2
                y[i][j] = j*dy-dy/2
                qc[i][j][0] = r[i][j]
                qc[i][j][1] = r[i][j]*u[i][j]
                qc[i][j][2] = r[i][j]*v[i][j]
                qc[i][j][3] = e[i][j]
                qf[i][j][0] = r[i][j]
                qf[i][j][1] = u[i][j]
                qf[i][j][2] = v[i][j]
                qf[i][j][3] = p[i][j]
        
        print('...done')
    
    def calmain(self):
        global qc

        print('   calculation...')

        for n in range(nt):
            print("    time step =",n+1)

            self.calrhs()

            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    qc[i][j] = qc[i][j] - dt*rhs[i][j]
            
            self.boundary()
            if np.mod(n,10)==0: self.output()
        
        print('   ...done')
        
    def boundary(self):
        global qc, qf
        for j in range(ny):
            qc[0][j] = 2.0*ql-qc[1][j]
            qc[nx-1][j] = qc[nx-2][j]
        
        for i in range(nx):
            qc[i][0] = qc[i][ny-2]
            qc[i][ny-1] = qc[i][1]
        
        for i in range(nx):
            for j in range(ny):
                qf[i][j][0] = qc[i][j][0]
                qf[i][j][1] = qc[i][j][1]/qc[i][j][0]
                qf[i][j][2] = qc[i][j][2]/qc[i][j][0]
                qf[i][j][3] = bet*(qc[i][j][3] \
                                   -0.5*qf[i][j][0]*(qf[i][j][1]**2+qf[i][j][2]**2))

    def calrhs(self):
        global rhs

        rhs = np.zeros_like(rhs)
        
        for m in range(2):
            if m == 0:
                id = 1
                jd = 0
                dl = dx
            else:
                id = 0
                jd = 1
                dl = dy

            self.fvs(m)

            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    rhs[i][j] += (fl[i][j] - fl[i-id][j-jd])/dl

    def fvs(self,m):
        global fl
        if m == 0:
            id = 1
            jd = 0
        else:
            id = 0
            jd = 1
        for i in range(0,nx-1):
            for j in range(0,ny-1):
              R,RI,GM,GA = self.Jacb(i,j,m)
              Ap = np.dot(np.dot(R,GM+GA),RI)

              R,RI,GM,GA = self.Jacb(i+id,j+jd,m)
              Am = np.dot(np.dot(R,GM-GA),RI)

              fl[i][j] = 0.5*(np.dot(Ap,qc[i][j])+np.dot(Am,qc[i+id][j+jd]))

    def Jacb(self,i,j,m):
        h = (qf[i][j][3]+qc[i][j][3])/qc[i][j][0]
        u = qf[i][j][1]
        v = qf[i][j][2]
        q = u**2+v**2
        c = np.sqrt(bet*(h-0.5*q))
        b2 = bet/c**2
        b1 = 0.5*b2*q

        if m == 0:
            R = np.array([[1.00,1.0,1.0,0.0],[u-c,u,u+c,0.0],
                          [v,v,v,1.0],[h-c*u,0.5*q,h+c*u,v]])
            RI = np.array([[0.5*(b1+u/c),-0.5*(1.0/c+b2*u),-0.5*b2*v,0.5*b2],
                          [1.0-b1,b2*u,b2*v,-b2],
                          [0.5*(b1-u/c),0.5*(1.0/c-b2*u),-0.5*b2*v,0.5*b2],
                          [-v,0.0,1.0,0.0]])
            GM = np.array([[u-c,0.0,0.0,0.0],[0.0,u,0.0,0.0],
                           [0.0,0.0,u+c,0.0],[0.0,0.0,0.0,u]])
            GA = np.array([[abs(u-c),0.0,0.0,0.0],[0.0,abs(u),0.0,0.0],
                           [0.0,0.0,abs(u+c),0.0],[0.0,0.0,0.0,abs(u)]])
        else:
            R = np.array([[1.0,1.0,1.0,0.0],[u,u,u,1.0],
                          [v-c,v,v+c,0.0],[h-c*v,0.5*q,h+c*v,u]])
            RI = np.array([[0.5*(b1+v/c),-0.5*b2*u,-0.5*(1.0/c+b2*v),0.5*b2],
                           [1.0-b1,b2*u,b2*v,-b2],
                           [0.5*(b1-v/c),-0.5*b2*u,0.5*(1.0/c-b2*v),0.5*b2],
                           [-u,1.0,0.0,0.0]])
            GM = np.array([[v-c,0.0,0.0,0.0],[0.0,v,0.0,0.0],
                           [0.0,0.0,v+c,0.0],[0.0,0.0,0.0,v]])
            GA = np.array([[abs(v-c),0.0,0.0,0.0],[0.0,abs(v),0.0,0.0],
                           [0.0,0.0,abs(v+c),0.0],[0.0,0.0,0.0,abs(v)]])

        return R,RI,GM,GA

    def output(self):
        r = np.empty((nx,ny))
        u = np.empty((nx,ny))
        p = np.empty((nx,ny))
        for i in range(nx):
            for j in range(ny):
                r[i][j] = qf[i][j][0]
                u[i][j] = qf[i][j][1]
                p[i][j] = qf[i][j][3]
        
        im1 = ax1.plot_wireframe(x,y,r)
        im2 = ax2.plot_wireframe(x,y,u)
        im3 = ax3.plot_wireframe(x,y,p)
        ims.append([im1,im2,im3])
    
    def imsave(self):
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        # plt.show()
        ani.save("sample2.gif", writer="pillow")

if __name__ == '__main__':
    proc = Euler()
    proc.main()
