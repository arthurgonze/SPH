#ifndef M_PI
    #define M_PI 3.14159265359
#endif
#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <string.h>
#include <time.h>
#include <omp.h>

using namespace Eigen;
using namespace std;

// particle data structure stores position, velocity full step, velocity half step, and force for integration
// stores density (rho) and pressure values for SPH
struct Particle
{
    Particle(float _x, float _y) : pos(_x, _y), vel(0.f, 0.f), vh(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f) {}
    Particle(float _x, float _y, float _vx, float _vy) : pos(_x, _y), vel(_vx, _vy), vh(_vx, _vy), f(0.f, 0.f), rho(0), p(0.f) {}
    Vector2d pos, vel, vh, f;
    float rho, p;
    int id;

    bool operator==(const Particle& rhs) const
    {
        return pos == rhs.pos && vel == rhs.vel;
    }

    bool operator<(const Particle& rhs) const
    {
        return pos(0) < rhs.pos(0) || pos(1) < rhs.pos(1);
    }
};

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al. solver parameters
const static Vector2d G(0.f, -10.f);   // external (gravitational) forces
const static float REST_DENS = 300.f;  // rest density
const static float GAS_CONST = 4000.f; // const for equation of state
const static float H = 16.f;		   // kernel radius
const static float HSQ = H * H;		   // radius^2 for optimization
const static float MASS = 2.5f;		   // assume all particles have the same mass
const static float VISC = 200.f;	   // viscosity constant
const static float DT = 0.0005f;	   // integration timestep
const static int N_TIME_STEPS = 2500;  // Number of timesteps

// smoothing kernels defined in Müller and their gradients adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

// simulation parameters
const static float EPS = H; // boundary epsilon
const static float BOUND_DAMPING = -0.5f;

// interaction
const static int MAX_PARTICLES = 10000;
const static int DAM_PARTICLES = 500;
const static int BLOCK_PARTICLES = 250;
const static int ADD_PARTICLES_EVERY = 120;
const static int ADD_PARTICLES_UNTIL = 1200;

// rendering projection parameters
const static float DOMAIN_SCALE = 1.5f;
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double DOMAIN_WIDTH = DOMAIN_SCALE * WINDOW_WIDTH; // 1.5*800=1200
const static double DOMAIN_HEIGHT = DOMAIN_SCALE * WINDOW_HEIGHT;// 1.5*600=900
const static double DOMAIN_X_LIM[2] = {EPS, DOMAIN_WIDTH - EPS};
const static double DOMAIN_Y_LIM[2] = {EPS, DOMAIN_HEIGHT - EPS};

// grid variables
const static float CELL_WIDTH = H+H; // 1200 / 16 = 75;
const static float CELL_HEIGHT = H+H; // 900 / 9 = 100;
const static int GRID_WIDTH_CELLS = DOMAIN_WIDTH/CELL_WIDTH; // 1200 / 75 = 16;
const static int GRID_HEIGHT_CELLS = DOMAIN_HEIGHT/CELL_HEIGHT; // 900 / 100 = 9;

// solver data
bool step = false;
bool advance_frame = false;
bool paused = true;
static int particles_in_sim = 0;
vector<Particle> grid[GRID_WIDTH_CELLS][GRID_HEIGHT_CELLS]; // MAP

int updates = 0;

// SPH
void AddParticles()
{
    unsigned int placed = 0;

    for (float y = (DOMAIN_HEIGHT*3)/4; y < DOMAIN_HEIGHT; y += H)
        for (float x = DOMAIN_WIDTH/3; x <= (DOMAIN_WIDTH*2)/3; x += H)
            if (placed < BLOCK_PARTICLES && particles_in_sim < MAX_PARTICLES)
            {
                int i = x/CELL_WIDTH;
                int j = y/CELL_HEIGHT;

                Particle p = Particle(x, y, 0, -1000);
                p.id = particles_in_sim;
                grid[i][j].push_back(p);
                particles_in_sim++;
                placed++;
            }
}

void InitSPH()
{
    for (float y = EPS; y < DOMAIN_HEIGHT - EPS; y += H)
        for (float pos = DOMAIN_WIDTH/3; pos <= (DOMAIN_WIDTH*2)/3; pos += H)
            if (particles_in_sim < DAM_PARTICLES)
            {
                float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // add a pos disturb
                int i = (pos + jitter)/CELL_WIDTH;
                int j = (y + jitter)/CELL_HEIGHT;

                Particle p = Particle(pos + jitter, y);
                p.id = particles_in_sim;
                grid[i][j].push_back(p);
                particles_in_sim++;
            }
            else
                return;
}

void Integrate()
{
    for(int i=0; i<GRID_WIDTH_CELLS; i++)
        for(int j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            if(grid[i][j].size()==0)
                continue;

            for (auto &p : grid[i][j])
            {
                // forward Euler integration
//                p.vel += DT * p.f / p.rho; // velocities + dt * forces / densities
//                p.pos += DT * p.vel; // positions + dt * velocities

                // leapfrog integration
                if(updates != 0)
                {
                    // vh[i] += a[i] * dt;
                    p.vh += (p.f / p.rho) * DT;
                    // v[i] = vh[i] + a[i] * dt / 2;
                    p.vel = p.vh + (p.f / p.rho) * (DT/2);
                    // x[i] += vh[i] * dt;
                    p.pos += p.vh * DT;
                }
                else
                {
                    // vh[i] = v[i] + a[i] * dt / 2;
                    p.vh = p.vel + (p.f / p.rho) * (DT/2);
                    // v[i] += a[i] * dt;
                    p.vel += (p.f / p.rho) * DT;
                    // x[i] += vh[i] * dt;
                    p.pos += p.vh * DT;
                }

                // enforce boundary conditions
                if (p.pos(0) < DOMAIN_X_LIM[0])
                {
                    p.vel(0) *= BOUND_DAMPING;
                    p.vh(0) *= BOUND_DAMPING;
                    p.pos(0) = DOMAIN_X_LIM[0];
                }
                if (p.pos(0) > DOMAIN_X_LIM[1])
                {
                    p.vel(0) *= BOUND_DAMPING;
                    p.vh(0) *= BOUND_DAMPING;
                    p.pos(0) = DOMAIN_X_LIM[1];
                }
                if (p.pos(1) < DOMAIN_Y_LIM[0])
                {
                    p.vel(1) *= BOUND_DAMPING;
                    p.vh(1) *= BOUND_DAMPING;
                    p.pos(1) = DOMAIN_Y_LIM[0];
                }
                if (p.pos(1) > DOMAIN_Y_LIM[1])
                {
                    p.vel(1) *= BOUND_DAMPING;
                    p.vh(1) *= BOUND_DAMPING;
                    p.pos(1) = DOMAIN_Y_LIM[1];
                }
            }
        }
}

void UpdateGridPositions()
{
    bool position_changed = false;
    for(int i=0; i<GRID_WIDTH_CELLS; i++)
        for(int j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            if(grid[i][j].size()==0)
                continue;

            vector<Particle> particles;
            for (auto &p : grid[i][j])
            {
                // update grid
                int new_i = p.pos(0)/CELL_WIDTH;
                if(new_i<0)
                    new_i=0;
                if(new_i >= GRID_WIDTH_CELLS)
                    new_i = GRID_WIDTH_CELLS-1;

                int new_j = p.pos(1)/CELL_HEIGHT;
                if(new_j<0)
                    new_j=0;
                if(new_j >= GRID_HEIGHT_CELLS)
                    new_j = GRID_HEIGHT_CELLS-1;

                if(new_i != i || new_j != j)
                {
                    grid[new_i][new_j].push_back(p);
                    position_changed = true;
                }
                else
                    particles.push_back(p);
            }
            grid[i][j].swap(particles);
            particles.clear();
        }
}

void ContParticles()
{
    int cont_particles = 0;
    for(int i=0; i<GRID_WIDTH_CELLS; i++)
        for(int j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            int particlesInCell = 0;
            if (grid[i][j].size() == 0)
                continue;
            for (auto &p : grid[i][j])
            {
                cont_particles++;
                particlesInCell++;
            }
//            cout << "Cell (" << i << ", " << j << "), have: " << particlesInCell << " particles" << endl;
        }
    particles_in_sim = cont_particles;
}

void ComputeDensityPressure()
{
    for(int i=0; i<GRID_WIDTH_CELLS; i++)
        for(int j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            if(grid[i][j].size()==0)
                continue;
            for (auto &pi : grid[i][j])
            {
                float rho = 0.f;

                //----------------------------------------
                int min_i = (pi.pos(0)-HSQ)/CELL_WIDTH;
                if(min_i < 0)
                    min_i = 0;
                //----------------------------------------
                int max_i = (pi.pos(0)+HSQ)/CELL_WIDTH;
                if(max_i >= GRID_WIDTH_CELLS)
                    max_i = GRID_WIDTH_CELLS-1;
                //----------------------------------------
                int min_j = (pi.pos(1)-HSQ)/CELL_HEIGHT;
                if(min_j < 0)
                    min_j = 0;
                //----------------------------------------
                int max_j = (pi.pos(1)+HSQ)/CELL_HEIGHT;
                if(max_j >= GRID_HEIGHT_CELLS)
                    max_j = GRID_HEIGHT_CELLS-1;
                //----------------------------------------
                for(int act_i = min_i; act_i <= max_i; act_i++)
                    for(int act_j = min_j; act_j <= max_j; act_j++)
                    {
                        if(grid[act_i][act_j].size()==0)
                            continue;
                        for (auto &pj: grid[act_i][act_j])
                        {
                            Vector2d rij = pj.pos - pi.pos;
                            float r2 = rij.squaredNorm();

                            if (r2 < HSQ)
                                rho += MASS * POLY6 * pow(HSQ - r2, 3.f); // this computation is symmetric
                        }
                    }
                if(rho != 0)
                    pi.rho = rho;
                if(pi.rho == 0)
                    pi.rho = REST_DENS;
                pi.p = GAS_CONST * (pi.rho - REST_DENS);
            }
        }
}

void ComputeForces()
{
    for(int i=0; i<GRID_WIDTH_CELLS; i++)
        for(int j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            if(grid[i][j].size()==0)
                continue;

            for (auto &pi : grid[i][j])
            {
                Vector2d fpress(0.f, 0.f);
                Vector2d fvisc(0.f, 0.f);

                //----------------------------------------
                int min_i = i-1;
                if(min_i < 0)
                    min_i = 0;
                //----------------------------------------
                int max_i = i+1;
                if(max_i >= GRID_WIDTH_CELLS)
                    max_i = GRID_WIDTH_CELLS-1;
                //----------------------------------------
                int min_j = j-1;
                if(min_j < 0)
                    min_j = 0;
                //----------------------------------------
                int max_j = j+1;
                if(max_j >= GRID_HEIGHT_CELLS)
                    max_j = GRID_HEIGHT_CELLS-1;
                //----------------------------------------
                for(int act_i = min_i; act_i <= max_i; act_i++)
                    for(int act_j = min_j; act_j <= max_j; act_j++)
                    {
                        if(grid[act_i][act_j].size()==0)
                            continue;
                        for (auto &pj: grid[act_i][act_j])
                        {
                            if (&pi == &pj)
                                continue;

                            Vector2d rij = pj.pos - pi.pos;
                            float r = rij.norm();

                            if (r < H)
                            {
                                fpress += -rij.normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
                                fvisc += VISC * MASS * (pj.vel - pi.vel) / pj.rho * VISC_LAP * (H - r);
                            }
                        }
                    }
                Vector2d fgrav = G * MASS / pi.rho;
                pi.f = fpress + fvisc + fgrav;
            }
        }
}

void Update()
{
    if(advance_frame || !paused)
    {
        cout << "****************************************************************" << endl;
        cout << "Frame: " << updates << ", Particles: " << particles_in_sim << endl;
        const clock_t update_time = clock();

        clock_t func_time = clock();
        if(updates != 0 && updates % ADD_PARTICLES_EVERY == 0 && updates <= ADD_PARTICLES_UNTIL)
        {
            func_time = clock();
            AddParticles();
//            std::cout << "Adding Particles time(s): " << float( clock () - func_time ) /  CLOCKS_PER_SEC << endl << endl;
        }

        func_time = clock();
        ComputeDensityPressure();
//        std::cout << "Compute Density time(s): " << float( clock () - func_time ) /  CLOCKS_PER_SEC << endl << endl;

        func_time = clock();
        ComputeForces();
//        std::cout << "Computing forces time(s): " << float( clock () - func_time ) /  CLOCKS_PER_SEC << endl << endl;

        func_time = clock();
        Integrate();
//        std::cout << "Integrate time(s): " << float( clock () - func_time ) /  CLOCKS_PER_SEC << endl << endl;

        func_time = clock();
        UpdateGridPositions();
//        std::cout << "Update grid time(s): " << float( clock () - func_time ) /  CLOCKS_PER_SEC << endl << endl;

        std::cout << "Frame time(s): " << float( clock () - update_time ) /  CLOCKS_PER_SEC << endl << endl << endl;

//        ContParticles();

        glutPostRedisplay();
        updates++;
        advance_frame = false;
    }
}

// OPEN GL FUNCTIONS
void InitGL()
{
    glClearColor(0.9f, 0.9f, 0.9f, 1);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(H / 2.f);
    glMatrixMode(GL_PROJECTION);
}

void DrawDomain()
{
    glColor3f(0.0,0.0,0.0);
    glLineWidth(H);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBegin(GL_POLYGON);
    glVertex2i(0,0);
    glVertex2i(DOMAIN_WIDTH,0);
    glVertex2i(DOMAIN_WIDTH,DOMAIN_HEIGHT);
    glVertex2i(0,DOMAIN_HEIGHT);
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glFlush();
}

void DrawBorders()
{
    glColor3f(1.0,0.0,0.0);
    glLineWidth(0.1);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBegin(GL_POLYGON);
    glVertex2i(DOMAIN_X_LIM[0],DOMAIN_Y_LIM[0]);
    glVertex2i(DOMAIN_X_LIM[1],DOMAIN_Y_LIM[0]);
    glVertex2i(DOMAIN_X_LIM[1],DOMAIN_Y_LIM[1]);
    glVertex2i(DOMAIN_X_LIM[0],DOMAIN_Y_LIM[1]);
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glFlush();
}

void DrawLine (float x1, float y1, float x2, float y2)
{
    glBegin (GL_LINES);
    glVertex2f (x1, y1);
    glVertex2f (x2, y2);
    glEnd();
}

void DrawGrid()
{
    glColor3ub (0, 255, 0);
    glLineWidth (1.0);

//    #pragma omp parallel for collapse(2)
    for (float i = DOMAIN_X_LIM[0]-EPS; i < DOMAIN_X_LIM[1]+EPS; i += CELL_WIDTH)
        for (float j = DOMAIN_Y_LIM[0]-EPS; j < DOMAIN_Y_LIM[1]+EPS; j += CELL_HEIGHT)
        {
            DrawLine(i, j, i + CELL_WIDTH, j);//horizontais
            DrawLine (i, j, i, j + CELL_HEIGHT);//verticais
        }
}

void DrawParticles()
{
    glColor4f(0.2f, 0.6f, 1.f, 1);
    glBegin(GL_POINTS);
    int i,j=0;
//    #pragma omp parallel for private(i,j) shared(grid)
    for(i=0; i<GRID_WIDTH_CELLS; i++)
        for(j=0; j<GRID_HEIGHT_CELLS;j++)
        {
            if(grid[i][j].size()==0)
                continue;
            for (auto &p: grid[i][j])
                {
                    glVertex2f(p.pos[0], p.pos[1]);
                }
        }
    glEnd();
}

void Render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glOrtho(0, DOMAIN_WIDTH, 0, DOMAIN_HEIGHT, 0, 1);

    DrawDomain();
//    DrawGrid();
    DrawBorders();
    DrawParticles();

    glutSwapBuffers();
}

void Keyboard(unsigned char c, __attribute__((unused)) int pos, __attribute__((unused)) int y)
{
    switch (c)
    {
        case 'a': // add particles
            AddParticles();
            break;
        case 'f': // advance frame
            advance_frame = true;
            break;
        case 'p': // advance frame
            paused = !paused;
            if(paused) advance_frame = false;
            break;
    }
}

// MAIN FUNCTION
int main(int argc, char **argv)
{
    srand(42);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInit(&argc, argv);
    glutCreateWindow("SPH 2D");
    glutDisplayFunc(Render);
    glutIdleFunc(Update);
    glutKeyboardFunc(Keyboard);

    InitGL();
    InitSPH();

    glutMainLoop();
    return 0;
}
