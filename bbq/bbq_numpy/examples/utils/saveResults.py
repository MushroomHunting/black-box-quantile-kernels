import numpy as np
import imageio
import json
import os


class ResultSaver:
    def __init__(self, settings_vars):
        self.expName = "{}_{}".format(
            settings_vars["datasetName"],
            "RBF" if settings_vars["basicRBF"] else "BBQ")
        if not settings_vars["basicRBF"]:
            self.expName = "{}_{}".format(
                self.expName,
                settings_vars["quantileParametrisation"])
        self.expName = "{}_{}".format(self.expName, settings_vars["nlOptAlgo"])

        # Ceate experiments dir if it doesnt exist
        exp_dir = os.path.join("dataDumps")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Create new directory for this experiment
        self.fileDir = os.path.join(exp_dir, self.expName)
        if not os.path.exists(self.fileDir):
            os.makedirs(self.fileDir)
        else:
            i = 2
            while os.path.exists(os.path.join(
                    exp_dir, "{}_rep{}".format(self.expName, i))):
                i += 1
            self.fileDir = os.path.join(exp_dir, "{}_rep{}".format(
                self.expName, i))
            os.makedirs(self.fileDir)

        print("###############################3")
        print("Saving results to {}".format(self.fileDir))

        # Save all variables. Could be handy...
        with open(os.path.join(self.fileDir, 'vars.json'), 'w') as fp:
            json.dump(settings_vars, fp, sort_keys=True, indent=4)

    def __save(self, file_name, data):
        np.savetxt(os.path.join(self.fileDir, "{}.csv".format(file_name)),
                   data, delimiter=",")

    def save_dataset(self, train_x, train_y, test_x, test_y):
        self.__save("train_x", train_x)
        self.__save("train_y", train_y)
        self.__save("test_x", test_x)
        self.__save("test_y", test_y)

    def save_data_fit(self, pred_train_x, pred_train_mean, pred_train_var,
                      pred_x, pred_mean, pred_var):
        self.__save("predTrainX", pred_train_x)
        self.__save("predTrainMean", pred_train_mean)
        self.__save("predTrainVar", pred_train_var)
        self.__save("predX", pred_x)
        self.__save("predMean", pred_mean)
        self.__save("predVar", pred_var)

    @staticmethod
    def save_pred_image(img_pixels):
        imageio.imwrite('predicted_image.png', img_pixels)

    def save_pdf(self, bbq_qf, interval_log=-3, n=10000):
        log_space = np.logspace(interval_log, -1, n)
        lin = np.hstack([log_space,
                         np.linspace(0.1 + 1 / n, 0.9 - 1 / n, n),
                         (1 - log_space)[::-1]])
        q_pts = bbq_qf(lin)
        pdf = (lin[:-1] - lin[1:]) / (q_pts[:-1] - q_pts[1:])
        lin_fixed = (q_pts[1:] + q_pts[:-1]) / 2.0
        self.__save("pdf_lin_x", lin_fixed)
        self.__save("pdf", pdf)

    def save_params_and_loss(self, all_params, losses):
        self.__save("loss_curve", losses)
        self.__save("all_params", all_params)

    def save_loss_surface(self, x, y, z):
        self.__save("loss_surface_x", x)
        self.__save("loss_surface_y", y)
        self.__save("loss_surface_z", z)

    def save_quantile(self, bbq_qf, points_x, points_y, n_pts=1000):
        lin = np.linspace(0, 1, n_pts)
        # ARD kernel case not handled
        if isinstance(bbq_qf, list):
            return
        else:  # Isotropic kernel cale. save single quantile
            quantile_curve = bbq_qf(lin)
            self.__save("quantile_pts_x", points_x)
            self.__save("quantile_pts_y", points_y)
            self.__save("quantile_curve", quantile_curve)
            self.__save("quantile_lin_x", lin)
