from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
from torch.utils.data import DataLoader
import utils
from uvit_datasets import get_dataset
import tempfile
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True,
                                     drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)

    nnet = utils.get_nnet(**config.nnet)
    nnet, test_dataset_loader = accelerator.prepare(nnet, test_dataset_loader)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator()

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode=t2i, mixed_precision={config.mixed_precision}')

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)
        return decode_large_batch(_z)

    def sample_fn(_n_samples, _context):
        assert _context.size(0) == _n_samples
        return dpm_solver_sample(_n_samples, config.sample.sample_steps, context=_context)


    # with tempfile.TemporaryDirectory() as temp_path:
    #     path = config.sample.path or temp_path
    #     if accelerator.is_main_process:
    #         os.makedirs(path, exist_ok=True)
    #     logging.info(f'Samples are saved in {path}')
    #     utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

    ### Start of updated code 

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        
        logging.info(f'Samples are saved in {path}')

#         # Define your 35 texts here - HUman hip
#         texts = [
# "After internal fixation of right femur",
# "Bilateral hip joint deformity",
# "Bone hyperplasia in the right acetabulum and bilateral sacroiliac joints",
# "Comminuted fracture of right femoral intertrochanteric",
# "Consider left inferior pubic ramus fracture",
# "Fracture of left ascending ramus of pubic bone",
# "Fracture of right ascending pubic ramus and ischial ramus",
# "Fracture of the left ascending and descending pubic bone",
# "Fracture of the upper right femur, no abnormalities found in the left hip joint",
# "Invisible sacral fissure with no abnormalities found in pelvis",
# "Left femoral head necrosis",
# "Left femoral intertrochanteric fracture",
# "Left femoral intertrochanteric fracture, high density shadow in left iliac region",
# "Left femoral intertrochanteric fracture with small trochanter tear",
# "left femoral neck fracture",
# "left hip joint dislocation",
# "left pubic bone  fracture",
# "Local cortical discontinuity at the right anterior superior iliac spine",
# "No abnormality in pelvis",
# "No obvious abnormalities  in the pelvis",
# "Poor cortical continuity of the right ascending pubic ramus",
# "Postoperative internal fixation of left femoral intertrochanteric fracture",
# "Postoperative internal fixation of right femoral  intertrochanteric fracture",
# "Postoperative internal fixation of right femoral neck base fracture",
# "Postoperative internal fixation of right femur",
# "Postoperative left hip joint replacement surgery",
# "Postoperative  old fracture of right calcaneus and internal fixation of right femoral neck fracture",
# "post right hip joint replacement surgery",
# "Removal of internal fixation device for intertrochanteric fracture of left femur",
# "Right femoral intertrochanteric linear fracture after internal fixation of left femoral intertrochanteric fracture",
# "Right femoral neck base fracture",
# "Right femoral neck fracture",
# "Right ilium fracture",
# "Short linear high-density shadows in the left iliac area and small pelvic area",
# "Suspected right acetabular fracture"
#         ]
        
### Medimgs generation

        texts = [
"MRI scan of the human brain in a non-demented individual, showing no signs of significant hippocampal atrophy or other structural abnormalities.",
"Microscopic image of benign acute lymphoblastic leukemia cells in a blood smear of human.",
"X-ray image of a human bone showing a visible fracture line.",
"MRI scan of the human brain showing glioma with irregular and infiltrative borders.",
"MRI scan of the human brain showing a neurocytoma, characterized by a well-circumscribed, rounded tumor in the brain's ventricles.",
"MRI scan showing schwannoma tumor, typically along cranial or spinal nerves in human.",
"Histological image of human breast tissue showing benign cell structures with no malignant features",
"Histological image showing cellular abnormalities indicative of disease in human tissue.",
"CT scan of a healthy human chest showing normal lung structure.",
"Clinical photograph showing redness and infection in a dog's conjunctiva.",
"Axial MRI scan showing normal brain anatomy without any pathological findings in human.",
"Clinical photographs showing various diseases affecting cattle, including skin lesions or swelling.",
"Endoscopic image showing polyps lifted and dyed for better visualization in the gastrointestinal tract in human.",
"Microscopic image of early-stage acute lymphoblastic leukemia cells in blood of human.",
"OCT image of the human retina revealing abnormal blood vessel growth beneath the retina, indicative of choroidal neovascularization.",
"Fundus photograph of the human eye showing optic nerve head cupping and rim thinning, consistent with glaucomatous damage.",
"Clinical photograph of the scalp showing redness, inflammation, and small pustules due to folliculitis in human.",
"Clinical photograph of gangrenous tissue showing blackened and necrotic skin in human.",
"Clinical photograph of gums showing redness and swelling characteristic of gingivitis in human.",
"Ultrasound image showing a healthy human kidney with normal size and function.",
"Clinical photograph of cattle eyes showing redness and swelling due to keratoconjunctivitis.",
"CT scan of the abdomen showing a mass indicative of human kidney tumor.",
"Clinical photograph showing abnormal skin and lesions on the leg of cattle.",
"Microscopic image showing fibrotic tissue replacing normal liver cells in human.",
"Microscopic image of liver tissue showing lipid accumulation within hepatocytes, consistent with fatty liver disease in human.",
"Clinical photograph of cattle showing raised, nodular lesions consistent with lumpy skin disease.",
"CT scan of the human chest showing a well-defined benign lung mass.",
"Clinical photograph of the human skin showing redness, inflammation, and swelling indicative of Lyme disease.",
"Microscopic image of human blood showing red blood cells infected with malaria parasites.",
"Microscopic image showing small to medium-sized lymphoid cells in mantle cell lymphoma in human.",
"Cytological smear image highlighting metaplastic cells with enlarged nuclei and irregular chromatin, often associated with cervical cancer in human.",
"Clinical photograph of the oral cavity showing an open sore indicative of mouth ulcer in human.",
"MRI sagittal scan highlighting hyperintense plaques along the spinal cord and brain's periventricular regions in human.",
"Clinical photograph of cattle showing nasal obstruction due to respiratory disease.",
"Clinical photograph showing normal facial features without Down syndrome characteristics in human.",
"CT scan of the human chest showing clear lung fields and no evidence of nodules or abnormal masses indicative of cancer.",
"X-ray of joints showing no signs of osteoarthritis, with smooth cartilage and normal joint spaces in human.",
"Clinical photograph of human legs showing no signs of varicose veins or swelling.",
"X-ray of the human chest showing normal lung and heart structures.",
"Endoscopic image showing a healthy pylorus with smooth and intact lining in human.",
"Ultrasound image of a human thyroid gland with normal size, texture, and function.",
"Clinical photograph of the human oral cavity showing abnormal growths consistent with oral cancer.",
"X-ray of joints showing joint space narrowing and bony growths typical of osteoarthritis in human.",
"Clinical photograph of throat showing inflammation and redness consistent with pharyngitis in human.",
"Human X-ray of the chest showing patchy infiltrates consistent with pneumonia.",
"X-ray of the shoulder showing the placement of a prosthetic implant in human.",
"Clinical photograph of a dog showing skin lesions, redness, or crusting around the eyes, indicative of ocular skin disease.",
"Clinical photograph showing human scalp infection with scaling and hair loss typical of Tinea Capitis.",
"Chest X-ray showing abnormalities such as cavities, nodules, or consolidation consistent with pulmonary tuberculosis in human.",
"Clinical photograph of cattle showing severe rough ring-like hyperkeratotic lesions."       
        ]

        # Iterate over the 50 texts
        for i, text in enumerate(texts):
            text_folder = text.replace(" ", "_").replace("/", "_")  # Sanitize folder name
            text_output_path = os.path.join(path, text_folder)
            os.makedirs(text_output_path, exist_ok=True)

            _context = next(context_generator)

            # Generate 10 images for each text
            utils.sample2dir(accelerator, text_output_path, 10, config.sample.mini_batch_size, 
                            lambda _n_samples: sample_fn(_n_samples, _context), dataset.unpreprocess)

            logging.info(f'Generated 10 images for text "{text}" in {text_output_path}')

    ### End of updated code

        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}, fid={fid}')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
