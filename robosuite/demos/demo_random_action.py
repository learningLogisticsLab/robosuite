from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from robosuite.utils.visual.VisualManager import VisualManager


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == 'bimanual':
            options["robots"] = 'Baxter'
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        #camera_names = ['agentview','agentview2']
    )
    env.reset()
    #env.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    eye = VisualManager(
        preprocessor_kwarg = dict(
            MODEL_ROOT = '/home/dizzyi/GNN/detectron/tutorial/output'
        ),
        imagesaver_kwarg = dict(
            save_mode = True,
            save_freq = 100,
            IMAGE_DIR = './imagesave'
        )
    )

    from PIL import Image
    import time
    # do visualization
    for i in range(1000):
        #delta = time.time()
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        feature_vectors = eye(obs['agentview_image'],env)

        '''

        img = Image.fromarray(img).rotate(180)
        segment = Image.fromarray(seg).rotate(180)

        objects = {}
        for i in seg.reshape(-1,3):
            name = env.sim.model.geom_id2name(i[1])
            objects[i[1]] = name
        
        for k, v in sorted(objects.items()): print(k, v.split("_") if v is not None else v)

        # objects [id] => name
        # ids list of M id 
        ids = np.unique(seg)
        #ids = np.array(list(filter(lambda id: objects[id] != None, ids)))

        # mask (256,256,1) with M ID
        _,mask,_ = np.split(seg,3,axis=2)

        # mask[np.newaxis]                           ==> ( 1, 256, 256, 1)
        # ids[:, np.newaxis, np.newaxis, np.newaxis] ==> ( M,   1,   1, 1)
        #                                                 L Broadcastable

        # masks  ==> (M, 256, 256, 1) 
        masks = ( mask[np.newaxis] == ids[:, np.newaxis, np.newaxis, np.newaxis]).squeeze().astype(np.uint8)
        #masks = np.array(list( filter( lambda m: m.sum() > 100, masks ) )) outdated
        masks = masks * 255



        img.save('./image.png')
        segment.save('./segment.png')
        
        for ind, msk in enumerate( masks ):
            seg_png = Image.fromarray(msk,mode='L').rotate(180)
            seg_png.save(f'./seg/{ids[ind]}-{objects[ids[ind]]}.png')
            print(ind)
        '''
        print('end loop')
        #print(time.time()-delta)
        #env.render()
