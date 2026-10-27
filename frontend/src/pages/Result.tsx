import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Film, Download, Trash2, ArrowLeft, Target, AlertTriangle, Camera } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { apiClient } from '../api/client';
import { PredictionDetail } from '../types';
import { getErrorMessage } from '../utils/errorHandler';

const gradeColors: Record<string, string> = {
  A: 'bg-green-100 text-green-700',
  B: 'bg-blue-100 text-blue-700',
  C: 'bg-amber-100 text-amber-700',
  'N/A': 'bg-gray-100 text-gray-700',
  Reject: 'bg-red-100 text-red-700',
};

const damageColors: Record<string, string> = {
  Damaged: 'bg-red-100 text-red-700',
  'Not Damaged': 'bg-green-100 text-green-700',
  'Needs Review': 'bg-yellow-100 text-yellow-700',
};

export function Result() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState<PredictionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const fetchPrediction = async () => {
      if (!id) return;
      try {
        const data = await apiClient.getPrediction(parseInt(id));
        setPrediction(data);
      } catch (err) {
        setError('Failed to load prediction');
      } finally {
        setLoading(false);
      }
    };
    fetchPrediction();
  }, [id]);

  const handleDelete = async () => {
    if (!id || !confirm('Are you sure you want to delete this prediction?')) return;

    setDeleting(true);
    try {
      await apiClient.deletePrediction(parseInt(id));
      window.location.href = '/history';
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDeleting(false);
    }
  };

  const handleDownload = async () => {
    if (!id || !prediction?.annotated_path) return;
    try {
      const blob = await apiClient.downloadAnnotated(parseInt(id));
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotated_${prediction.file_name}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Failed to download:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error || !prediction) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">{error || 'Prediction not found'}</p>
        <Link to="/history" className="mt-4 inline-block text-primary-600 hover:text-primary-700">
          Back to History
        </Link>
      </div>
    );
  }

  const finalCounts = {
    A: prediction.detection_boxes.filter((box) => box.grade === 'A').length,
    B: prediction.detection_boxes.filter((box) => box.grade === 'B').length,
    C: prediction.detection_boxes.filter((box) => box.grade === 'C').length,
    NA: prediction.detection_boxes.filter((box) => box.grade === 'N/A').length,
    Reject: prediction.detection_boxes.filter((box) => box.grade === 'Reject').length,
  };

  const provisionalCounts = {
    A: prediction.detection_boxes.filter((box) => box.provisional_grade === 'A').length,
    B: prediction.detection_boxes.filter((box) => box.provisional_grade === 'B').length,
    C: prediction.detection_boxes.filter((box) => box.provisional_grade === 'C').length,
    Reject: prediction.detection_boxes.filter((box) => box.provisional_grade === 'Reject').length,
  };

  const provisionalOnly = prediction.detection_boxes.some(
    (box) => box.grade === 'N/A' && box.provisional_grade && box.provisional_grade !== 'N/A'
  );

  const summaryItems = [
    { key: 'A', label: 'Grade A', count: finalCounts.A, color: 'bg-green-50', textColor: 'text-green-700', iconColor: 'text-green-600' },
    { key: 'B', label: 'Grade B', count: finalCounts.B, color: 'bg-blue-50', textColor: 'text-blue-700', iconColor: 'text-blue-600' },
    { key: 'C', label: 'Grade C', count: finalCounts.C, color: 'bg-amber-50', textColor: 'text-amber-700', iconColor: 'text-amber-600' },
    { key: 'NA', label: 'N/A', count: finalCounts.NA, color: 'bg-gray-50', textColor: 'text-gray-700', iconColor: 'text-gray-600' },
    { key: 'Reject', label: 'Reject', count: finalCounts.Reject, color: 'bg-red-50', textColor: 'text-red-700', iconColor: 'text-red-600' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            to="/history"
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 truncate max-w-md">
            {prediction.file_name}
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => navigate('/upload?mode=camera')}>
            <Camera className="w-4 h-4 mr-2" />
            Next Tray
          </Button>
          <Button onClick={() => navigate('/upload')} variant="outline">
            Upload File
          </Button>
          {prediction.annotated_path && (
            <Button onClick={handleDownload} variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Download Annotated
            </Button>
          )}
          <Button onClick={handleDelete} variant="outline" isLoading={deleting}>
            <Trash2 className="w-4 h-4 mr-2" />
            Delete
          </Button>
        </div>
      </div>

      {provisionalOnly && (
        <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-amber-900">Provisional grades only</p>
            <p className="text-sm text-amber-800 mt-1">
              Cleanliness grading is unavailable, so every final grade is N/A.
              Provisional grades use damage and size only and must not be treated as final.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardContent className="p-0">
            <div className="bg-gray-100 rounded-lg overflow-hidden" style={{ height: '400px' }}>
              {prediction.file_type === 'image' ? (
                prediction.annotated_path ? (
                  <img
                    src={`/uploads/${prediction.annotated_path.split(/[/\\]/).pop()}`}
                    alt={prediction.file_name}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-center text-gray-500 px-4">
                    <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-yellow-500" />
                    <p className="text-lg font-medium text-gray-700">No annotated image</p>
                    <p className="text-sm text-gray-500 mt-1">
                      {prediction.total_detections > 0
                        ? `But detected ${prediction.total_detections} egg(s) (${prediction.not_damaged_count} intact, ${prediction.damaged_count} damaged)`
                        : 'Could not detect any eggs in this image'}
                    </p>
                    <p className="text-xs text-gray-400 mt-2 break-all">Original: {prediction.file_name}</p>
                  </div>
                )
              ) : (
                <div className="text-center text-gray-500">
                  <Film className="w-12 h-12 mx-auto mb-2" />
                  <p>Video preview</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Status</span>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    prediction.status === 'completed' ? 'bg-green-100 text-green-700' :
                    prediction.status === 'failed' ? 'bg-red-100 text-red-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {prediction.status}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Processing Time</span>
                  <span className="text-sm font-medium">{prediction.processing_time_ms}ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Total Detections</span>
                  <span className="text-sm font-medium">{prediction.total_detections}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Needs Review</span>
                  <span className="text-sm font-medium">{prediction.needs_review_count}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Tray ID</span>
                  <span className="text-xs font-mono text-gray-500 truncate max-w-[140px]">
                    {prediction.tray_id || '-'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Final Grades</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {summaryItems.map((item) => (
                  <div key={item.key} className={`flex items-center justify-between p-3 ${item.color} rounded-lg`}>
                    <div className="flex items-center gap-2">
                      <Target className={`w-5 h-5 ${item.iconColor}`} />
                      <span className={`text-sm ${item.textColor}`}>{item.label}</span>
                    </div>
                    <span className={`text-lg font-bold ${item.textColor}`}>
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-gray-100">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Provisional (damage + size)
                </p>
                <div className="flex flex-wrap gap-2 mt-2 text-xs">
                  <span className="px-2 py-1 rounded bg-green-50 text-green-700">A: {provisionalCounts.A}</span>
                  <span className="px-2 py-1 rounded bg-blue-50 text-blue-700">B: {provisionalCounts.B}</span>
                  <span className="px-2 py-1 rounded bg-amber-50 text-amber-700">C: {provisionalCounts.C}</span>
                  <span className="px-2 py-1 rounded bg-red-50 text-red-700">Reject: {provisionalCounts.Reject}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {prediction.detection_boxes.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-10 text-center">
            <AlertTriangle className="h-10 w-10 text-amber-500" />
            <div>
              <p className="font-medium text-gray-800">No eggs detected</p>
              <p className="mt-1 text-sm text-gray-500">
                Capture the next tray or try a clearer image.
              </p>
            </div>
            <Button onClick={() => navigate('/upload?mode=camera')}>
              <Camera className="w-4 h-4 mr-2" />
              Next Tray
            </Button>
          </CardContent>
        </Card>
      )}

      {prediction.detection_boxes.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Per-Egg Results ({prediction.detection_boxes.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Egg</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Final</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Provisional</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Size</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Weight</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Damage</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Cleanliness</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Confidence</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Review</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.detection_boxes.map((box, index) => (
                    <tr key={box.id} className="border-b border-gray-100 align-top">
                      <td className="py-2 px-3 text-sm text-gray-600">
                        #{box.egg_id ?? index + 1}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-1 text-xs rounded-full ${gradeColors[box.grade || 'N/A']}`}>
                          {box.grade || 'N/A'}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-1 text-xs rounded-full ${gradeColors[box.provisional_grade || 'N/A']}`}>
                          {box.provisional_grade || 'N/A'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-sm text-gray-600">
                        <div>{box.size || 'Unknown'}</div>
                        {box.geometry?.length_mm && box.geometry?.breadth_mm && (
                          <div className="text-xs text-gray-400">
                            {box.geometry.length_mm.toFixed(1)} x {box.geometry.breadth_mm.toFixed(1)} mm
                          </div>
                        )}
                        <div className="text-xs text-gray-400">
                          {box.geometry?.quality || 'unknown'}
                        </div>
                      </td>
                      <td className="py-2 px-3 text-sm text-gray-600">
                        {box.weight_g ? `${box.weight_g}g` : '-'}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          damageColors[box.damage_status || 'Needs Review']
                        }`}>
                          {box.damage_status || 'Needs Review'}
                        </span>
                        {box.confidence.damage != null && (
                          <div className="text-xs text-gray-400 mt-1">
                            {(box.confidence.damage * 100).toFixed(1)}%
                          </div>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700">
                          {box.cleanliness_status || 'Unknown'}
                        </span>
                        {box.confidence.cleanliness != null && (
                          <div className="text-xs text-gray-400 mt-1">
                            {(box.confidence.cleanliness * 100).toFixed(1)}%
                          </div>
                        )}
                      </td>
                      <td className="py-2 px-3 text-sm text-gray-600">
                        {(box.confidence.detection * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-xs">
                        {box.requires_review ? (
                          <div>
                            <span className="px-2 py-1 rounded-full bg-yellow-100 text-yellow-700">
                              Review
                            </span>
                            {box.review_reason && (
                              <div className="text-gray-400 mt-1 max-w-[180px] break-words">
                                {box.review_reason}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="px-2 py-1 rounded-full bg-green-100 text-green-700">
                            OK
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
