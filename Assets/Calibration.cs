using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Assets;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;

public class Calibration {
    // NB used Emgu usage example from Mappity https://github.com/benkuper/Mappity

    private Camera _mainCamera;
    private static int _numImages = 1;
    private MCvTermCriteria _termCriteria;
    private CALIB_TYPE _flags;
    private Size _size;

    public Calibration(Camera mainCamera)
    {
        _mainCamera = mainCamera;
        _termCriteria = new MCvTermCriteria();
        _flags = CALIB_TYPE.CV_CALIB_USE_INTRINSIC_GUESS | CALIB_TYPE.CV_CALIB_FIX_K1 | CALIB_TYPE.CV_CALIB_FIX_K2 | CALIB_TYPE.CV_CALIB_FIX_K3 | CALIB_TYPE.CV_CALIB_FIX_K4 | CALIB_TYPE.CV_CALIB_FIX_K5 | CALIB_TYPE.CV_CALIB_ZERO_TANGENT_DIST;
        _size = new Size(Screen.width, Screen.height);
    }

    public double calibrateFromCorrespondences(List<Vector3> imagePoints, List<Vector3> worldPoints)
    {
        PointF[][] imagePointsCvMat = PutImagePointsIntoArray(imagePoints, imagePoints.Count);
        MCvPoint3D32f[][] worldPointsCvMat = PutObjectPointsIntoArray(worldPoints, imagePoints.Count, _numImages);
        IntrinsicCameraParameters intrinsic = CreateIntrinsicGuess();
        ExtrinsicCameraParameters[] extrinsics;
        var calibrationError = CameraCalibration.CalibrateCamera(worldPointsCvMat, imagePointsCvMat, _size, intrinsic, _flags, _termCriteria, out extrinsics);
        ApplyCalibrationToUnityCamera(intrinsic, extrinsics[0]);
        return calibrationError;
    }

    private void ApplyCalibrationToUnityCamera(IntrinsicCameraParameters intrinsic, ExtrinsicCameraParameters extrinsics)
    {
        Matrix<double> rotationInverse = flipZAxis(extrinsics.RotationVector.RotationMatrix).Transpose(); // transpose is same as inverse for rotation matrix
        Matrix<double> transFinal = (rotationInverse * -1) * extrinsics.TranslationVector;
        _mainCamera.projectionMatrix = LoadProjectionMatrix((float)intrinsic.IntrinsicMatrix[0, 0], (float)intrinsic.IntrinsicMatrix[1, 1], (float)intrinsic.IntrinsicMatrix[0, 2], (float)intrinsic.IntrinsicMatrix[1, 2]);
        ApplyTranslationAndRotationToCamera(transFinal, RotationConversion.RotationMatrixToEulerZXY(rotationInverse));
    }

    private static Matrix<double> flipZAxis(Matrix<double> rotationMatrix)
    {
        var LHSflipBackMatrix = new Matrix<double>(3, 3);
        LHSflipBackMatrix.SetIdentity();
        LHSflipBackMatrix[2, 2] = -1.0; // flip only the z axis (right handed to left handed coordinate system flip)
        return rotationMatrix * LHSflipBackMatrix;;
    }

    private void ApplyTranslationAndRotationToCamera(Matrix<double> translation, Rotation r)
    {
        _mainCamera.transform.position = new Vector3((float)translation[0, 0], (float)translation[1, 0], (float)translation[2, 0]);
        _mainCamera.transform.eulerAngles = new Vector3((float)r.X, (float)r.Y, (float)r.Z);
    }

    private Matrix4x4 LoadProjectionMatrix(float fx, float fy, float cx, float cy)
    {
        // https://github.com/kylemcdonald/ofxCv/blob/88620c51198fc3992fdfb5c0404c37da5855e1e1/libs/ofxCv/src/Calibration.cpp
        float w = _mainCamera.pixelWidth;
        float h = _mainCamera.pixelHeight;
        float nearDist = _mainCamera.nearClipPlane;
        float farDist = _mainCamera.farClipPlane;

        return MakeFrustumMatrix(
            nearDist * (-cx) / fx, nearDist * (w - cx) / fx,
            nearDist * (cy) / fy, nearDist * (cy - h) / fy,
            nearDist, farDist);
    }

    private Matrix4x4 MakeFrustumMatrix(float left, float right,
                                        float bottom, float top,
                                        float zNear, float zFar)
    {
        // https://github.com/openframeworks/openFrameworks/blob/master/libs/openFrameworks/math/ofMatrix4x4.cpp
        // note transpose of ofMatrix4x4 wr.t OpenGL documentation, since the OSG use post multiplication rather than pre.
        // NB this has been transposed here from the original openframeworks code

        float A = (right + left) / (right - left);
        float B = (top + bottom) / (top - bottom);
        float C = -(zFar + zNear) / (zFar - zNear);
        float D = -2.0f * zFar * zNear / (zFar - zNear);

        var persp = new Matrix4x4();
        persp[0, 0] = 2.0f * zNear / (right - left);
        persp[1, 1] = 2.0f * zNear / (top - bottom);
        persp[2, 0] = A;
        persp[2, 1] = B;
        persp[2, 2] = C;
        persp[2, 3] = -1.0f;
        persp[3, 2] = D;

        var rhsToLhs = new Matrix4x4();
        rhsToLhs[0, 0] = 1.0f;
        rhsToLhs[1, 1] = -1.0f; // Flip Y (RHS -> LHS)
        rhsToLhs[2, 2] = 1.0f;
        rhsToLhs[3, 3] = 1.0f;

        return rhsToLhs * persp.transpose; // see comment above
    }

    private IntrinsicCameraParameters CreateIntrinsicGuess()
    {
        double height = (double) _size.Height;
        double width = (double) _size.Width;

        // from https://docs.google.com/spreadsheet/ccc?key=0AuC4NW61c3-cdDFhb1JxWUFIVWpEdXhabFNjdDJLZXc#gid=0
        // taken from http://www.neilmendoza.com/projector-field-view-calculator/
        float hfov = 91.2705674249382f;
        float vfov = 59.8076333281726f;

        double fx = (double)((float)width / (2.0f * Mathf.Tan(0.5f * hfov * Mathf.Deg2Rad)));
        double fy = (double)((float)height / (2.0f * Mathf.Tan(0.5f * vfov * Mathf.Deg2Rad)));

        double cy = height / 2.0;
        double cx = width / 2.0;

        var intrinsics = new IntrinsicCameraParameters(); 
        intrinsics.IntrinsicMatrix[0, 0] = fx;
        intrinsics.IntrinsicMatrix[0, 2] = cx;
        intrinsics.IntrinsicMatrix[1, 1] = fy;
        intrinsics.IntrinsicMatrix[1, 2] = cy;
        intrinsics.IntrinsicMatrix[2, 2] = 1;

        return intrinsics;
    }

    private MCvPoint3D32f[][] PutObjectPointsIntoArray(List<Vector3> _objectPositions, int pointsCount, int ImageNum)
    {
        var objectPoints = new MCvPoint3D32f[1][];
        objectPoints[0] = new MCvPoint3D32f[pointsCount];

        for (int i = 0; i < pointsCount; i++)
            objectPoints[0][i] = new MCvPoint3D32f(_objectPositions[i].x, _objectPositions[i].y, _objectPositions[i].z * -1);

        return objectPoints;
    }

    private PointF[][] PutImagePointsIntoArray(List<Vector3> _imagePositions, int pointsCount)
    {
        var imagePoints = new PointF[1][];
        imagePoints[0] = new PointF[pointsCount];

        for (int i = 0; i < _imagePositions.Count; i++)
            imagePoints[0][i] = new PointF(_imagePositions[i].x, _imagePositions[i].y);

        return imagePoints;
    }
}