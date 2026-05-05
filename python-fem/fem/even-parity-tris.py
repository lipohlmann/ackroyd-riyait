import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

from scipy.sparse import lil_matrix, kron, csr_matrix
from scipy.sparse.linalg import gmres, LinearOperator, spilu, bicgstab, lgmres

from scipy.integrate import dblquad, quad

from linear_triangles import *


def Mu(theta, phi):
    return np.cos(phi)


def Eta(theta, phi):
    return np.sin(phi) * np.cos(theta)


def K_mu_mu(vertices, j, n, det_jacobian):
    result = dblquad(
        K_mu_mu_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def K_mu_mu_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    mu_sqrd = Mu(theta, phi) ** 2
    return (
        mu_sqrd
        * LinearBasis(xi, eta, j)
        * LinearBasis(xi, eta, n)
        * np.sin(phi)
        * det_jacobian
    )


def K_eta_eta(vertices, j, n, det_jacobian):
    result = dblquad(
        K_eta_eta_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def K_eta_eta_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    eta_sqrd = Eta(theta, phi) ** 2
    return (
        eta_sqrd
        * LinearBasis(xi, eta, j)
        * LinearBasis(xi, eta, n)
        * np.sin(phi)
        * det_jacobian
    )


def K_mu_eta(vertices, j, n, det_jacobian):
    result = dblquad(
        K_mu_eta_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def K_mu_eta_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    eta_mu = Eta(theta, phi) * Mu(theta, phi)
    return (
        eta_mu
        * LinearBasis(xi, eta, j)
        * LinearBasis(xi, eta, n)
        * np.sin(phi)
        * det_jacobian
    )


def M_xx(x_gradients, i, m, triangle_area):
    return x_gradients[i] * x_gradients[m] * triangle_area


def M_yy(y_gradients, i, m, triangle_area):
    return y_gradients[i] * y_gradients[m] * triangle_area


def M_xy(x_gradients, y_gradients, i, m, triangle_area):
    return x_gradients[i] * y_gradients[m] * triangle_area


def M_yx(x_gradients, y_gradients, i, m, triangle_area):
    return y_gradients[i] * x_gradients[m] * triangle_area


def B_mu(j, n, vertices, det_jacobian):
    result = dblquad(
        B_mu_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def B_mu_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    mu = abs(Mu(theta, phi))
    return (
        mu
        * LinearBasis(xi, eta, j)
        * LinearBasis(xi, eta, n)
        * np.sin(phi)
        * det_jacobian
    )


def B_eta(j, n, vertices, det_jacobian):
    result = dblquad(
        B_eta_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def B_eta_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    y_cosine = abs(Eta(theta, phi))
    return (
        y_cosine
        * LinearBasis(xi, eta, j)
        * LinearBasis(xi, eta, n)
        * np.sin(phi)
        * det_jacobian
    )


def K(j, n, vertices, det_jacobian):
    result = dblquad(
        K_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(j, n, vertices, det_jacobian),
    )[0]
    return result


def K_integrand(eta, xi, j, n, vertices, det_jacobian):
    theta, phi = ReferenceToReal(vertices, xi, eta)
    return (
        LinearBasis(xi, eta, j) * LinearBasis(xi, eta, n) * np.sin(phi) * det_jacobian
    )


def M_matrix(verices):
    area = TriangleArea(verices)
    return (area / 6) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)


def A(n, vertices, det_jacobian):
    result = dblquad(
        A_integrand,
        0,
        1,
        lambda xi: 0,
        lambda xi: 1 - xi,
        args=(n, vertices, det_jacobian),
    )[0]
    return result


def A_integrand(eta, xi, n, vertices, det_jacobian):
    _, phi = ReferenceToReal(vertices, xi, eta)
    return LinearBasis(xi, eta, n) * det_jacobian * np.sin(phi)


def S(xi, eta, vertices, det_jacobian, scattering_cross_sections):
    xs = scattering_cross_sections / 4 / np.pi
    return xs * M_matrix(vertices)


def Q(m, vertices, det_jacobian, Qs):
    return (
        Qs[m]
        * dblquad(
            Q_integrand,
            0,
            1,
            lambda xi: 0,
            lambda xi: 1 - xi,
            args=(m, vertices, det_jacobian),
        )[0]
    )


def Q_integrand(eta, xi, m, vertices, det_jacobian):
    return LinearBasis(xi, eta, m) * det_jacobian


# =============================================================================
# Read meshes
# =============================================================================
spatial_mesh = meshio.read("gmsh/square-source.msh")
angle_mesh = meshio.read("gmsh/solid-angle.msh")

# --- Spatial mesh connectivity ---
triangles = []
triangle_phys = []
lines = []
line_phys = []

for block, phys in zip(spatial_mesh.cells, spatial_mesh.cell_data["gmsh:physical"]):
    if block.type == "triangle":
        triangles.append(block.data)
        triangle_phys.append(phys)
    elif block.type == "line":
        lines.append(block.data)
        line_phys.append(phys)

triangles = np.vstack(triangles)
triangle_phys = np.concatenate(triangle_phys)
lines = np.vstack(lines)
line_phys = np.concatenate(line_phys)

west_edges = lines[line_phys == 13]
north_edges = lines[line_phys == 14]
east_edges = lines[line_phys == 15]
south_edges = lines[line_phys == 16]

# --- Material properties (per element) ---
Sigma_t = np.zeros(len(triangles))
Sigma_s = np.zeros(len(triangles))
Qs = np.zeros(len(triangles))

# material = 1, void = 2, source = 3
Sigma_t[triangle_phys == 1] = 0.8
Sigma_t[triangle_phys == 2] = 1e-5
Sigma_t[triangle_phys == 3] = 0.8

Sigma_s[triangle_phys == 1] = 0.0
Sigma_s[triangle_phys == 2] = 0.0
Sigma_s[triangle_phys == 3] = 0.0

# volumetric source strengths
Qs[triangle_phys == 1] = 0
Qs[triangle_phys == 2] = 0
Qs[triangle_phys == 3] = 6.4

points = spatial_mesh.points[:, :2]
n_space = len(points)

# =============================================================================
# Spatial global matrix assembly
# =============================================================================
M_global = lil_matrix((n_space, n_space))
# sigma_t-weighted stiffness matrices  (1/sigma_t * M_xx, etc.)
M_xx_sig = lil_matrix((n_space, n_space))
M_yy_sig = lil_matrix((n_space, n_space))
M_xy_sig = lil_matrix((n_space, n_space))
M_yx_sig = lil_matrix((n_space, n_space))
# sigma_t-weighted mass matrix for collision term
M_eff = lil_matrix((n_space, n_space))
# sigma_s/4pi-weighted mass matrix for scattering term
M_scat = lil_matrix((n_space, n_space))
# boundary matrices
B_x_global = lil_matrix((n_space, n_space))
B_y_global = lil_matrix((n_space, n_space))
b_space = np.zeros(n_space)

for e, tri in enumerate(triangles):
    verts = points[tri]  # (3, 2)
    area = TriangleArea(verts)
    x_grads, y_grads = BuildBasisDxDy(verts)

    sigma_t = Sigma_t[e]
    sigma_s = Sigma_s[e]
    inv_sigt = (1.0 / sigma_t) if sigma_t > 0.0 else 0.0

    M_loc = M_matrix(verts)

    for i_local, i_global in enumerate(tri):
        for m_local, m_global in enumerate(tri):
            val_xx = M_xx(x_grads, i_local, m_local, area)
            val_yy = M_yy(y_grads, i_local, m_local, area)
            val_xy = M_xy(x_grads, y_grads, i_local, m_local, area)
            val_yx = M_yx(x_grads, y_grads, i_local, m_local, area)
            M_loc_ij = M_loc[i_local, m_local]

            M_global[i_global, m_global] += M_loc_ij
            # FIX #4: per-element 1/sigma_t weighting (not global mean)
            M_xx_sig[i_global, m_global] += inv_sigt * val_xx
            M_yy_sig[i_global, m_global] += inv_sigt * val_yy
            M_xy_sig[i_global, m_global] += inv_sigt * val_xy
            M_yx_sig[i_global, m_global] += inv_sigt * val_yx
            # sigma_t * M  for collision operator
            M_eff[i_global, m_global] += sigma_t * M_loc_ij
            # sigma_s/(4pi) * M  for scattering operator
            M_scat[i_global, m_global] += (sigma_s / (4.0 * np.pi)) * M_loc_ij

    # Source vector:  integral of T_i over element = area/3
    q = Qs[e]
    if q != 0.0:
        for i_local, i_global in enumerate(tri):
            b_space[i_global] += q * area / 3.0
# compute mesh centroid once, before the loop
mesh_centroid = points.mean(axis=0)

# --- Boundary assembly (vacuum BC contribution) ---
VACUUM_TAGS = {14, 15}
REFLECT_TAGS = {13, 16}

for edge, tag in zip(lines, line_phys):
    n1, n2 = edge
    p1, p2 = points[n1], points[n2]
    edge_vec = p2 - p1
    edge_mid = 0.5 * (p1 + p2)
    normal = np.array([edge_vec[1], -edge_vec[0]])
    normal = normal / np.linalg.norm(normal)

for edge, tag in zip(lines, line_phys):
    if tag in REFLECT_TAGS:
        continue

    n1, n2 = edge
    p1, p2 = points[n1], points[n2]
    edge_vec = p2 - p1
    length = np.linalg.norm(edge_vec)

    # both candidate normals
    normal = np.array([edge_vec[1], -edge_vec[0]])
    normal = normal / np.linalg.norm(normal)

    # ensure normal points AWAY from the mesh centroid
    edge_mid = 0.5 * (p1 + p2)
    if np.dot(normal, edge_mid - mesh_centroid) < 0:
        normal = -normal

    nx, ny = normal

    B_edge = (length / 6.0) * np.array([[2, 1], [1, 2]])
    nodes = [n1, n2]
    for i_local, i_global in enumerate(nodes):
        for m_local, m_global in enumerate(nodes):
            B_x_global[i_global, m_global] += B_edge[i_local, m_local] * abs(nx)
            B_y_global[i_global, m_global] += B_edge[i_local, m_local] * abs(ny)

# Convert to CSR
M_global = M_global.tocsr()
M_xx_sig = M_xx_sig.tocsr()
M_yy_sig = M_yy_sig.tocsr()
M_xy_sig = M_xy_sig.tocsr()
M_yx_sig = M_yx_sig.tocsr()
M_eff = M_eff.tocsr()
M_scat = M_scat.tocsr()
B_x_global = B_x_global.tocsr()
B_y_global = B_y_global.tocsr()

# =============================================================================
# Angle mesh connectivity
# =============================================================================
angle_points = angle_mesh.points[:, :2]
angle_tris = []
for block in angle_mesh.cells:
    if block.type == "triangle":
        angle_tris.append(block.data)
angle_tris = np.vstack(angle_tris)
n_angle = len(angle_points)

# =============================================================================
# Angular global matrix assembly
# =============================================================================
K_mu_mu_g = lil_matrix((n_angle, n_angle))
K_eta_eta_g = lil_matrix((n_angle, n_angle))
K_mu_eta_g = lil_matrix((n_angle, n_angle))
K_g = lil_matrix((n_angle, n_angle))
B_mu_g = lil_matrix((n_angle, n_angle))
B_eta_g = lil_matrix((n_angle, n_angle))
A_vec = np.zeros(n_angle)

for tri in angle_tris:
    verts = angle_points[tri]
    detJ = DeterminantJacobian(verts)
    for j_local, j_global in enumerate(tri):
        for n_local, n_global in enumerate(tri):
            K_mu_mu_g[j_global, n_global] += K_mu_mu(verts, j_local, n_local, detJ)
            K_eta_eta_g[j_global, n_global] += K_eta_eta(verts, j_local, n_local, detJ)
            K_mu_eta_g[j_global, n_global] += K_mu_eta(verts, j_local, n_local, detJ)
            K_g[j_global, n_global] += K(j_local, n_local, verts, detJ)
            B_mu_g[j_global, n_global] += B_mu(j_local, n_local, verts, detJ)
            B_eta_g[j_global, n_global] += B_eta(j_local, n_local, verts, detJ)
    for n_local, n_global in enumerate(tri):
        A_vec[n_global] += A(n_local, verts, detJ)

K_mu_mu_g = K_mu_mu_g.tocsr()
K_eta_eta_g = K_eta_eta_g.tocsr()
K_mu_eta_g = K_mu_eta_g.tocsr()
K_g = K_g.tocsr()
B_mu_g = B_mu_g.tocsr()
B_eta_g = B_eta_g.tocsr()

# C_vec == A_vec for isotropic angular basis
C_vec = A_vec.copy()

# =============================================================================
# Global system matrix:
#
#   [B_mu ⊗ B_x  +  B_eta ⊗ B_y]  Ψ
# + [K_μμ ⊗ (1/σt)M_xx  +  K_ηη ⊗ (1/σt)M_yy
#    + K_με ⊗ ((1/σt)M_xy + (1/σt)M_yx)]  Ψ
# + [K ⊗ σt·M]  Ψ
# - [outer(C,A) ⊗ (σs/4π)·M]  Ψ          <-- scattering moves to LHS
# = Q_vec ⊗ A_vec
#
# Kronecker convention: kron(K_angle, M_space)
# => vector ordering: psi_flat[j * n_space + i]  (angle-major)
# =============================================================================

# Scattering angular matrix: outer product C_vec (row) ⊗ A_vec (col)
S_angle_mat = csr_matrix(np.outer(C_vec, A_vec))  # (n_angle, n_angle)

A_global = (
    kron(B_mu_g, B_x_global)
    + kron(B_eta_g, B_y_global)
    + kron(K_mu_mu_g, M_xx_sig)
    + kron(K_eta_eta_g, M_yy_sig)
    + kron(K_mu_eta_g, M_xy_sig + M_yx_sig)
    + kron(K_g, M_eff)
    - kron(S_angle_mat, M_scat)  # subtract scattering from LHS
)

# =============================================================================
# RHS:  b = Q_vec ⊗ A_vec
#
# Q_vec[i] = q_i * area_i / 3   (already in b_space, divided by 4π)
# Per the weak form: source = (Q / 4π) * ∫T_m dτ * ∫A_n dΩ
# So the spatial part is b_space_4pi[i] = q_i * area_i / (3 * 4π)
# and the full RHS is kron(ones_angle, b_space_4pi) weighted by A_vec,
# which in kron(angle, space) ordering is:
#   b_full[j * n_space + i] = A_vec[j] * b_space_4pi[i]
# => np.kron(A_vec, b_space_4pi)
# =============================================================================
b_space_4pi = b_space / (4.0 * np.pi)  # apply 1/4π factor here
b_full = np.kron(A_vec, b_space_4pi)

# =============================================================================
# Diagnostics
# =============================================================================
print("triangle_phys unique:", np.unique(triangle_phys))
print("Qs unique:", np.unique(Qs))
print("nonzero Qs count:", np.count_nonzero(Qs))
print("b_space max:", np.max(np.abs(b_space)))
print("b_full norm:", np.linalg.norm(b_full))
print("A_vec sum:", np.sum(A_vec))
print("System size:", A_global.shape)

# =============================================================================
# Solve
# =============================================================================
psi_flat, info = lgmres(A_global, b_full, atol=1e-10, maxiter=10000)
if info != 0:
    print(f"WARNING: GMRES did not converge (info={info})")
else:
    print("GMRES converged.")

# FIX #1/#2: reshape as (n_angle, n_space)  — angle-major ordering
psi_mat = psi_flat.reshape((n_angle, n_space))

# Scalar flux: phi_i = sum_j  A_j * psi_{ji}
phi = 4 * A_vec @ psi_mat  # shape (n_space,)

print("phi max:", np.max(phi))
print("phi min:", np.min(phi))

# =============================================================================
# Plot
# =============================================================================
x = points[:, 0]
y = points[:, 1]
triang = mtri.Triangulation(x, y, triangles=triangles)

plt.figure(figsize=(8, 6))
tpc = plt.tripcolor(triang, phi, cmap="magma", shading="gouraud")
plt.colorbar(tpc, label=r"Scalar Flux $\varphi$")
plt.triplot(triang, linewidth=0.3, color="k", alpha=0.4)
plt.title("Scalar Flux on Spatial Mesh Nodes")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("scalar_flux.png", dpi=200)

plt.figure(figsize=(8, 6))
phi_plot = np.clip(phi, a_min=1e-10, a_max=None)
tpc = plt.tripcolor(
    triang,
    phi_plot,
    cmap="magma",
    norm=LogNorm(vmin=phi_plot.min(), vmax=phi_plot.max()),
)
plt.colorbar(tpc, label=r"Scalar Flux $\varphi$")
plt.triplot(triang, linewidth=0.3, color="k", alpha=0.4)
plt.title("Scalar Flux on Spatial Mesh Nodes")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.savefig("scalar_flux_log.png", dpi=200)
