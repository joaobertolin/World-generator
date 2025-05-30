import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_closing
from skimage.morphology import disk
from noise import pnoise2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

class WorldGenerator:
    def __init__(self, size=1000):
        self.size = size
        self.height_map = np.zeros((size, size))
        self.plate_map = np.zeros((size, size), dtype=int)
        self.stress_map = np.zeros((size, size))
        self.land_mask = np.zeros((size, size), dtype=bool)
        self.water_mask = np.zeros((size, size), dtype=bool)
        self.ice_mask = np.zeros((size, size), dtype=bool)
        
        # Configurações padrão
        self.params = {
            'plates': 12,
            'continents': 5,
            'islands': 300,
            'water_percent': 70,
            'ice_percent': 10,
            'min_ocean': -4000,
            'max_land': 4500
        }
        
        # Paleta de cores para oceanos e terra
        self.colors = {
            'deep_ocean': '#0b1c3a',    # >3000m
            'ocean': '#1a4b7a',         # >1500m
            'shallow_ocean': '#2a7fbc', # >300m
            'coastal': '#6baed6',       # >0m
            'shore': '#e8d8b9',
            'lowland': '#8cbb5e',
            'highland': '#567d46',
            'mountain': '#3d5a40',
            'snow': '#ffffff',
            'ice': '#d4f1f9'
        }
        
    def generate_tectonic_plates(self, num_plates):
        print("Gerando placas tectônicas...")
        # Geração de pontos iniciais
        points = np.random.rand(num_plates, 2) * self.size
        
        # Aplicar relaxamento de Lloyd (3 iterações)
        for _ in range(3):
            vor = Voronoi(points)
            regions = [vor.regions[i] for i in vor.point_region]
            new_points = []
            for region in regions:
                if -1 not in region and len(region) > 0:
                    polygon = [vor.vertices[i] for i in region]
                    centroid = np.mean(polygon, axis=0)
                    new_points.append(centroid)
            points = np.array(new_points)
        
        # Criar mapa de placas
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
        plate_map = np.zeros((self.size, self.size), dtype=int)
        
        for i in range(self.size):
            for j in range(self.size):
                distances = np.sqrt((points[:,0] - i)**2 + (points[:,1] - j)**2)
                plate_map[j, i] = np.argmin(distances)
        
        # Gerar vetores de movimento para cada placa
        plate_vectors = np.random.rand(num_plates, 2) * 2 - 1
        
        # Calcular mapa de tensões tectônicas
        stress_map = np.zeros((self.size, self.size))
        for i in range(num_plates):
            plate_mask = (plate_map == i)
            if not np.any(plate_mask):
                continue
            
            # Encontrar bordas da placa
            edges = binary_closing(plate_mask) ^ plate_mask
            
            # Calcular tensão nas bordas
            for j in range(num_plates):
                if i == j:
                    continue
                neighbor_mask = (plate_map == j)
                if not np.any(neighbor_mask):
                    continue
                    
                # Calcular tensão baseada na direção relativa das placas
                vec_diff = plate_vectors[i] - plate_vectors[j]
                tension = np.dot(vec_diff, vec_diff)**0.5
                
                # Aplicar tensão nas áreas de borda
                stress_map[np.logical_and(edges, neighbor_mask)] = tension
        
        # Suavizar mapa de tensões
        stress_map = gaussian_filter(stress_map, sigma=5)
        
        self.plate_map = plate_map
        self.stress_map = stress_map
        return plate_map, stress_map

    def generate_continents(self, num_continents):
        print("Gerando continentes...")
        # Selecionar placas para continentes
        plate_ids = np.unique(self.plate_map)
        continent_plates = np.random.choice(plate_ids, size=min(num_continents, len(plate_ids)), replace=False)
        
        continent_mask = np.zeros((self.size, self.size), dtype=bool)
        noise_scale = 0.005
        
        for plate_id in continent_plates:
            plate_mask = (self.plate_map == plate_id)
            
            # Gerar ruído Perlin dentro da placa
            plate_noise = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    if plate_mask[i, j]:
                        plate_noise[i, j] = pnoise2(i*noise_scale, j*noise_scale, octaves=6)
            
            # Normalizar e limiarizar o ruído
            plate_noise = (plate_noise - plate_noise.min()) / (plate_noise.max() - plate_noise.min())
            continent_threshold = np.random.uniform(0.4, 0.6)
            continent_plate_mask = (plate_noise > continent_threshold)
            
            # Adicionar ao continente principal
            continent_mask = np.logical_or(continent_mask, continent_plate_mask)
        
        # Operações morfológicas para melhorar a forma
        continent_mask = binary_closing(continent_mask, disk(3))
        continent_mask = gaussian_filter(continent_mask.astype(float), sigma=2) > 0.5
        
        self.land_mask = continent_mask
        return continent_mask

    def generate_islands(self, num_islands):
        print("Gerando ilhas...")
        # Gerar pontos usando amostragem de Poisson
        ocean_mask = ~self.land_mask
        distance = distance_transform_edt(ocean_mask)
        
        # Ajustar densidade com base na distância da costa
        probability_map = np.clip(distance / 50, 0, 1)
        probability_map = probability_map**2 * 0.1
        
        # Gerar pontos de ilhas
        island_points = []
        attempts = 0
        max_attempts = num_islands * 10
        
        with tqdm(total=num_islands, desc="Criando ilhas") as pbar:
            while len(island_points) < num_islands and attempts < max_attempts:
                x, y = np.random.randint(0, self.size, 2)
                if ocean_mask[x, y] and np.random.rand() < probability_map[x, y]:
                    island_points.append((x, y))
                    pbar.update(1)
                attempts += 1
        
        # Criar ilhas
        island_mask = np.zeros((self.size, self.size), dtype=bool)
        noise_scale = 0.02
        
        for x, y in island_points:
            # Tamanho aleatório para ilhas
            size = max(5, int(np.random.exponential(10)))
            island_value = np.random.rand()
            
            for i in range(max(0, x-size), min(self.size, x+size)):
                for j in range(max(0, y-size), min(self.size, y+size)):
                    if not ocean_mask[i, j]:
                        continue
                        
                    dist = np.sqrt((x-i)**2 + (y-j)**2)
                    if dist > size:
                        continue
                    
                    # Forma baseada em ruído e distância
                    noise_val = pnoise2(i*noise_scale, j*noise_scale, octaves=2)
                    island_strength = (1 - dist/size) * (0.5 + noise_val*0.5)
                    
                    if island_strength > 0.3:
                        island_mask[i, j] = True
        
        # Adicionar ilhas ao mapa de terra
        self.land_mask = np.logical_or(self.land_mask, island_mask)
        return island_mask

    def generate_terrain_height(self):
        print("Gerando mapa de altura...")
        # Configurar elevações base
        height_map = np.zeros((self.size, self.size))
        
        # 1. Configurar placas oceânicas e continentais
        ocean_plates = []
        continent_plates = []
        
        for plate_id in np.unique(self.plate_map):
            plate_area = np.sum(self.plate_map == plate_id)
            if plate_area < self.size**2 * 0.1:  # Placas pequenas são oceânicas
                ocean_plates.append(plate_id)
            else:
                continent_plates.append(plate_id)
        
        # 2. Aplicar elevações base
        for i in range(self.size):
            for j in range(self.size):
                plate_id = self.plate_map[i, j]
                
                if plate_id in ocean_plates:
                    # Bacias oceânicas profundas
                    height_map[i, j] = np.random.uniform(self.params['min_ocean'], -2000)
                else:
                    # Áreas continentais
                    height_map[i, j] = np.random.uniform(-500, 500)
        
        # 3. Adicionar efeito de tensão tectônica
        height_map += self.stress_map * 1000
        
        # 4. Aplicar ruído para terreno natural
        print("Aplicando ruído de terreno...")
        terrain_noise = np.zeros((self.size, self.size))
        noise_scale = 0.005
        
        for i in tqdm(range(self.size)):
            for j in range(self.size):
                terrain_noise[i, j] = pnoise2(
                    i*noise_scale, 
                    j*noise_scale, 
                    octaves=6, 
                    persistence=0.5
                )
        
        # Normalizar ruído e aplicar
        terrain_noise = (terrain_noise - terrain_noise.min()) / (terrain_noise.max() - terrain_noise.min())
        height_map += terrain_noise * 3000
        
        # 5. Limitar alturas conforme parâmetros
        height_map = np.clip(
            height_map, 
            self.params['min_ocean'], 
            self.params['max_land']
        )
        
        # 6. Ajustar nível do mar
        water_percent = self.params['water_percent'] / 100
        sorted_heights = np.sort(height_map.flatten())
        sea_level_index = int(len(sorted_heights) * water_percent)
        sea_level = sorted_heights[sea_level_index]
        
        # 7. Criar máscara de água
        self.water_mask = height_map < sea_level
        self.height_map = height_map
        self.sea_level = sea_level
        
        return height_map, sea_level

    def generate_ice(self):
        print("Gerando camadas de gelo...")
        ice_mask = np.zeros((self.size, self.size), dtype=bool)
        target_percent = self.params['ice_percent'] / 100
        
        # 1. Gelo polar (latitudes extremas)
        pole_height = int(self.size * 0.15)
        north_pole = np.s_[:pole_height, :]
        south_pole = np.s_[-pole_height:, :]
        
        # 2. Selecionar áreas terrestres em regiões polares
        for region in [north_pole, south_pole]:
            land_in_region = self.land_mask[region] & (self.height_map[region] > 0)
            ice_mask[region][land_in_region] = True
        
        # 3. Gelo em montanhas altas
        mountain_ice = (self.height_map > 3000) & self.land_mask
        ice_mask = np.logical_or(ice_mask, mountain_ice)
        
        # 4. Ajustar para porcentagem desejada
        current_percent = np.sum(ice_mask) / (self.size * self.size)
        
        # Se precisar de mais gelo, expandir para áreas mais baixas
        if current_percent < target_percent:
            needed_area = int(target_percent * self.size**2) - np.sum(ice_mask)
            candidates = self.land_mask & ~ice_mask & (self.height_map > 1000)
            
            # Ordenar candidatos por latitude (prioridade para áreas polares)
            lat_priority = np.abs(np.arange(self.size)[:, None] - self.size/2
            candidate_scores = np.where(candidates, lat_priority, -1)
            
            # Selecionar áreas mais prioritárias
            flat_indices = np.argsort(candidate_scores.flatten())[::-1]
            selected = np.unravel_index(
                flat_indices[:needed_area], 
                (self.size, self.size)
            )
            
            ice_mask[selected] = True
        
        self.ice_mask = ice_mask
        return ice_mask

    def visualize_world(self):
        print("Renderizando mundo...")
        # Criar mapa de cores
        cmap = LinearSegmentedColormap.from_list('world_colors', [
            self.colors['deep_ocean'],        # < -3000m
            self.colors['ocean'],             # -3000m a -1500m
            self.colors['shallow_ocean'],     # -1500m a -300m
            self.colors['coastal'],           # -300m a 0m
            self.colors['shore'],             # 0m a 50m
            self.colors['lowland'],           # 50m a 500m
            self.colors['highland'],          # 500m a 2000m
            self.colors['mountain'],          # 2000m a 3500m
            self.colors['snow']               # >3500m
        ], N=256)
        
        # Aplicar cores baseadas na altitude
        norm_heights = (self.height_map - self.params['min_ocean']) / (self.params['max_land'] - self.params['min_ocean'])
        colored_world = cmap(norm_heights)
        
        # Aplicar gelo
        colored_world[self.ice_mask] = mcolors.to_rgba(self.colors['ice'])
        
        # Aplicar bordas continentais
        shore_mask = binary_closing(self.water_mask, disk(1)) ^ self.water_mask
        colored_world[shore_mask] = mcolors.to_rgba(self.colors['shore'])
        
        # Plotar
        plt.figure(figsize=(12, 12))
        plt.imshow(colored_world)
        plt.title(f"Mundo Gerado: {self.params['continents']} continentes, {self.params['islands']} ilhas")
        plt.axis('off')
        
        # Adicionar legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color=self.colors['deep_ocean'], label='Oceano Profundo (>3000m)'),
            Patch(color=self.colors['ocean'], label='Oceano (1500-3000m)'),
            Patch(color=self.colors['shallow_ocean'], label='Oceano Raso (300-1500m)'),
            Patch(color=self.colors['coastal'], label='Águas Costeiras (0-300m)'),
            Patch(color=self.colors['lowland'], label='Terras Baixas'),
            Patch(color=self.colors['highland'], label='Planaltos'),
            Patch(color=self.colors['mountain'], label='Montanhas'),
            Patch(color=self.colors['ice'], label='Gelo'),
        ]
        
        plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0))
        plt.tight_layout()
        plt.savefig('mundo_gerado.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_world(self, **kwargs):
        # Atualizar parâmetros
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        # Ordem de geração
        self.generate_tectonic_plates(self.params['plates'])
        self.generate_continents(self.params['continents'])
        self.generate_islands(self.params['islands'])
        self.generate_terrain_height()
        self.generate_ice()
        self.visualize_world()

# Exemplo de uso
if __name__ == "__main__":
    generator = WorldGenerator(size=1000)
    
    # Personalizar parâmetros (opcional)
    custom_params = {
        'plates': 15,
        'continents': 6,
        'islands': 500,
        'water_percent': 75,
        'ice_percent': 15,
        'min_ocean': -4500,
        'max_land': 5000
    }
    
    generator.generate_world(**custom_params)
