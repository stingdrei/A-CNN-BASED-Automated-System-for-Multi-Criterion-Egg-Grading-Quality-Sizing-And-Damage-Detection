import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Film, Download, Trash2, ArrowLeft, Target, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { apiClient } from '../api/client';
import { PredictionDetail } from '../types';

export function Result() {
  const { id } = useParams<{ id: string }>();
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
        <div className="flex items-center gap-2">
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
                    {prediction.file_path && (
                      <p className="text-xs text-gray-400 mt-2 break-all">Original: {prediction.file_path.split(/[/\\]/).pop()}</p>
                    )}
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
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(() => {
                  const gradeCounts = {
                    AA: prediction.detection_boxes.filter(b => b.grade === 'AA').length,
                    A: prediction.detection_boxes.filter(b => b.grade === 'A').length,
                    B: prediction.detection_boxes.filter(b => b.grade === 'B').length,
                    NA: prediction.detection_boxes.filter(b => b.grade === 'N/A').length,
                    Reject: prediction.detection_boxes.filter(b => b.grade === 'Reject').length,
                  };
                  
                  const gradeItems = [
                    { key: 'AA', label: 'Grade AA', count: gradeCounts.AA, color: 'bg-green-50', textColor: 'text-green-700', iconColor: 'text-green-600' },
                    { key: 'A', label: 'Grade A', count: gradeCounts.A, color: 'bg-blue-50', textColor: 'text-blue-700', iconColor: 'text-blue-600' },
                    { key: 'B', label: 'Grade B', count: gradeCounts.B, color: 'bg-amber-50', textColor: 'text-amber-700', iconColor: 'text-amber-600' },
                    { key: 'NA', label: 'N/A', count: gradeCounts.NA, color: 'bg-gray-50', textColor: 'text-gray-700', iconColor: 'text-gray-600' },
                    { key: 'Reject', label: 'Reject', count: gradeCounts.Reject, color: 'bg-red-50', textColor: 'text-red-700', iconColor: 'text-red-600' },
                  ];
                  
                  return gradeItems.map((item) => (
                    <div key={item.key} className={`flex items-center justify-between p-3 ${item.color} rounded-lg`}>
                      <div className="flex items-center gap-2">
                        <Target className={`w-5 h-5 ${item.iconColor}`} />
                        <span className={`text-sm ${item.textColor}`}>{item.label}</span>
                      </div>
                      <span className={`text-lg font-bold ${item.textColor}`}>
                        {item.count}
                      </span>
                    </div>
                  ));
                })()}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {prediction.detection_boxes.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Detections ({prediction.detection_boxes.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Grade</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Class</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Size</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Weight</th>
                    <th className="text-left py-2 px-3 text-sm font-medium text-gray-500">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.detection_boxes.map((box) => {
                    const gradeColors: Record<string, string> = {
                      AA: 'bg-green-100 text-green-700',
                      A: 'bg-blue-100 text-blue-700',
                      B: 'bg-amber-100 text-amber-700',
                      'N/A': 'bg-gray-100 text-gray-700',
                      Reject: 'bg-red-100 text-red-700',
                    };
                    return (
                      <tr key={box.id} className="border-b border-gray-100">
                        <td className="py-2 px-3">
                          <span className={`px-2 py-1 text-xs rounded-full ${gradeColors[box.grade || 'N/A']}`}>
                            {box.grade || 'N/A'}
                          </span>
                        </td>
                        <td className="py-2 px-3">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            box.class_name === 'damaged' 
                              ? 'bg-red-100 text-red-700' 
                              : 'bg-green-100 text-green-700'
                          }`}>
                            {box.class_name}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-sm text-gray-600">
                          {box.size_category || '-'}
                        </td>
                        <td className="py-2 px-3 text-sm text-gray-600">
                          {box.weight_g ? `${box.weight_g}g` : '-'}
                        </td>
                        <td className="py-2 px-3 text-sm text-gray-600">
                          {(box.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}