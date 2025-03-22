from opencood.utils.transformation_utils import get_pairwise_transformation
from v2i_calib.v2x_calib.corresponding.BoxesMatch import BoxesMatch
from v2i_calib.v2x_calib.corresponding.CorrespondingDetector import CorrespondingDetector
from v2i_calib.v2x_calib.search.Matches2Extrinsics import Matches2Extrinsics
from v2i_calib.v2x_calib.reader.BBox3d import BBox3d
from v2i_calib.v2x_calib.utils import implement_T_3dbox_object_list


def convert_opencoodCornerslist_to_boxesObjectlist(corners_list):
    """
    Args:
        corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

    Returns:
        boxes3d_list: List[List[v2i_calib.BBox3d]]
            [[N_1, BBox3d], ..., [N_cav1, BBox3d]]
    """
    boxes3d_list = []
    for cav in range(len(corners_list)):
        box3d_list = []
        for i in range(corners_list[cav].shape[0]):
            box_8_3 = corners_list[cav][i]
            box3d_list.append(BBox3d('default_type', box_8_3))
        boxes3d_list.append(box3d_list)
    return boxes3d_list



def get_matches_from_boxes_list_trueT(boxes3d_list, pairwise_Tmat_list):
    """
    Args:
        boxes3d_list: List[List[v2i_calib.BBox3d]]
            [[N_1, BBox3d], ..., [N_cav1, BBox3d]]

        pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j

    Returns:
        matches_list: List[List[match]]
            [[N_1, 2], ..., [N_cav1, 2]]
    """
    matches_list = []
    for cav1 in range(len(boxes3d_list)):
        for cav2 in range(cav1+1, len(boxes3d_list)):
            boxes1 = boxes3d_list[cav1]
            boxes2 = boxes3d_list[cav2]
            Tmat = pairwise_Tmat_list[cav1, cav2]
            converted_boxes1 = implement_T_3dbox_object_list(Tmat, boxes1)
            matches = CorrespondingDetector(converted_boxes1, boxes2).get_matches()
            matches_list.append(list(matches))
    return matches_list


def get_matches_from_boxes_list_v2icalib(boxes3d_list):
    """
    Args:
        boxes3d_list: List[List[v2i_calib.BBox3d]]
            [[N_1, BBox3d], ..., [N_cav1, BBox3d]]

    Returns:
        matches_list: List[List[match]]
            [[N_1, 2], ..., [N_cav1, 2]]
    """
    
    pass

def get_matches_from_boxes_list_nn(boxes3d_list):
    """
    Args:
        boxes3d_list: List[List[v2i_calib.BBox3d]]
            [[N_1, BBox3d], ..., [N_cav1, BBox3d]]

    Returns:
        matches_list: List[List[match]]
            [[N_1, 2], ..., [N_cav1, 2]]
    """
    
    pass


def get_extrinsic_from_matches(boxes3d_list, matches_list):
    """
    Args:
        boxes3d_list: List[List[v2i_calib.BBox3d]]
            [[N_1, BBox3d], ..., [N_cav1, BBox3d]]

        matches_list: List[List[match]]
            [[N_1, 2], ..., [N_cav1, 2]]

    Returns:
        extrinsic_list: List[extrinsic]
            [N_cav1, 6]
    """
    extrinsic_list = []
    for cav1 in range(len(matches_list)):
        for cav2 in range(cav1+1, len(matches_list)):
            matches = matches_list[cav1]
            common_boxes1 = [boxes3d_list[cav1][match[0]] for match in matches]
            common_boxes2 = [boxes3d_list[cav2][match[1]] for match in matches]
            extrinsic = Matches2Extrinsics(common_boxes1, common_boxes2).get_combined_extrinsic()
            extrinsic_list.append(extrinsic)
    return extrinsic_list


def cal_init_pose(pred_corners_list, agent_pose_list, method):
    """
    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        agent_pose_list:
            [N_cav1, 6], in degree"

        method: str
            "v2icalib_origin" or "v2icalib_nn"
    """
    
    boxes3d_list = convert_opencoodCornerslist_to_boxesObjectlist(pred_corners_list)

    max_cav = len(pred_corners_list)
    pairwise_Tmat_list = get_pairwise_transformation(agent_pose_list, max_cav, False)
    

    if method == "v2icalib_origin":
        matches_list = get_matches_from_boxes_list_v2icalib(boxes3d_list)
    elif method == "v2icalib_nn":
        matches_list = get_matches_from_boxes_list_nn(boxes3d_list)
    elif method == "v2icalib_trueT":
        matches_list = get_matches_from_boxes_list_trueT(boxes3d_list, pairwise_Tmat_list)
    else:
        raise ValueError("method should be 'v2icalib_origin' or 'v2icalib_nn' or 'v2icalib_trueT'")

    extrinsic_list = get_extrinsic_from_matches(boxes3d_list, matches_list)

    return extrinsic_list
    