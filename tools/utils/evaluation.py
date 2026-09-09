import numpy as np
from sklearn.metrics import roc_auc_score
import open3d as o3d
import os
import torch
class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(aff_pred, aff_gt):

    '''
    contact:[B, 6890, 1]
    affordance:[B, 2048, 1]
    '''
    # dist_matrix = np.load('smpl_models/smpl_neutral_geodesic_dist.npy')
    # dist_matrix = torch.tensor(dist_matrix)

   
    aff_pred = aff_pred.cpu().detach().numpy()
    aff_gt = aff_gt.cpu().detach().numpy()

    AUC_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))
    IOU_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))

    SIM_matrix = np.zeros(aff_gt.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    num = aff_gt.shape[0]

    for b in range(num):
        #sim
        SIM_matrix[b] = SIM(aff_pred[b], aff_gt[b])

        #AUC_IOU
        aff_t_true = (aff_gt[b] >= 0.5).astype(int)
        aff_p_score = aff_pred[b]

        if np.sum(aff_t_true) == 0:
            AUC_aff[b] = np.nan
            IOU_aff[b] = np.nan
        else:
            try:
                auc_aff = roc_auc_score(aff_t_true, aff_p_score)
                AUC_aff[b] = auc_aff
            except ValueError:
                #print(pts_path[b])
                AUC_aff[b] = np.nan

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (aff_p_score >= thre).astype(int)
                intersect = np.sum(p_mask & aff_t_true)
                union = np.sum(p_mask | aff_t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU_aff[b] = aiou

    AUC_ = np.nanmean(AUC_aff)
    IOU_ = np.nanmean(IOU_aff)
    SIM_ = np.mean(SIM_matrix)

    return AUC_, IOU_, SIM_

def precision_recall_f1score(pred, gt):
    """
    Compute precision, recall, and f1
    """
    precision_avg = 0
    recall_avg = 0
    f1_avg = 0
    for b in range(gt.shape[0]):
        tp_num = gt[b, pred[b,:,0] >= 0.5, 0].sum()
        precision_denominator = (pred[b, :, 0] >= 0.5).sum()
        recall_denominator = (gt[b, :, 0]).sum()

        precision_ = tp_num / (precision_denominator + 1e-10)
        recall_ = tp_num / (recall_denominator + 1e-10)
        f1_ = 2 * precision_ * recall_ / (precision_ + recall_ + 1e-10)

        precision_avg += precision_
        recall_avg += recall_
        f1_avg += f1_

    return precision_avg / gt.shape[0], recall_avg / gt.shape[0], f1_avg / gt.shape[0]

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def mean_per_vertex_position_error(pred, gt):
    """
    Compute MPVPE
    """
    import torch
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).detach().mean(dim=-1).cpu().numpy()
        return error

def generate_proxy_sphere(center, pts_path):

    radii = {"Bag": 0.192, "Mug": 0.094, "Chair":0.455, "Guitar":0.394, "Bottle": 0.140, "Backpack": 0.265, "Tennisracket": 0.298, "Skateboard": 0.375, "Surfboard":0.687,
    "Earphone":0.132, "Suitcase":0.332, "Vase":0.197, "Umbrella":0.372, "Scissors":0.179, "Motorcycle":0.710, "Baseballbat":0.325, "Bed":1.154, "Knife":0.173,
    "Keyboard":0.217, "Bowl":0.132, "Bicycle":0.675}
    Object = (pts_path.split('/')[-1]).split('_')[0]
    radius = radii[Object]
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)
    sphere_mesh.compute_vertex_normals()
    sphere_mesh.translate(center)

    return sphere_mesh

def get_affordance_label(str, label):

    affordance_list = ['grasp', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 'pour', 
                'move', 'pull', 'wear', 'press', 'cut', 'stab', 'ride', 'play', 'carry']
    cut_str = str.split('_')
    affordance = cut_str[-2]
    index = affordance_list.index(affordance)

    label = label[:, index]
    
    return label

def visual_pred(img_path, affordance_pred, GT_path, results_folder):
    with open(GT_path,'r') as f:
        coordinates = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]

        gt_point = o3d.geometry.PointCloud()
        gt_point.points = o3d.utility.Vector3dVector(points_coordinates)

        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(points_coordinates)

        reference_color = np.array([255, 0, 0])
        back_color = np.array([190, 190, 190])

        pred_color = np.zeros((2048,3))
        for i, pred in enumerate(affordance_pred):
            scale_i = pred
            pred_color[i] = (reference_color-back_color) * scale_i + back_color

        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
        object = GT_path.split('/')[2]
        affordance_type = img_path.split('_')[-2]
        num = (GT_path.split('_')[-1]).split('.')[0]

        results_folder = results_folder + object + '/'
        if os.path.exists(results_folder) == False:
            os.makedirs(results_folder)
        pred_file = results_folder + object + '_' + affordance_type + '_' + num + '_Pred' + '.ply'
        o3d.io.write_point_cloud(pred_file, pred_point)
        f.close()